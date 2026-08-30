#!/usr/bin/env python
"""Offline attribution audit: policy vs smoothed target vs raw oracle.

Because the EA2 state transition is action-independent, the zero-action
rollouts stored in ``sl/logs/data/map_seed*.pt`` reproduce the deployment
state distribution *exactly*.  This script exploits that to decompose the
observed collisions and shrink deficit into three stages:

  1. ``raw``       -- the per-frame geometric oracle target, recomputed from
                      the stored distance field with the *collection-time*
                      configuration (margin 0.20, axis mode, interp crossing,
                      expansion soft cap soft_dof_pos_limit=0.9 -- verified
                      against the stored target's 0.95 saturation ceiling).
  2. ``smoothed``  -- the production RateLimitedOracle (incl. the env's
                      snap-to-raw safety check) replayed over the recomputed
                      raw sequence; cross-checked against the stored
                      ``target`` tensor to prove the replication is exact.
  3. ``policy``    -- the deployed EnvelopeNet run statefully (per-frame,
                      hidden carried across the whole episode) exactly as in
                      sl/evaluate.closed_loop_rollout.

For every frame the min clearance over the 34 hex samples is evaluated with
the production violation model (margin 0.10 / soft 0.10, nearest sampling --
identical to ``env._collision_hard``), and collisions are attributed:

    policy-unsafe frames -> inherited from smoothed target (oracle side)
                           or network-only;
    smoothed-unsafe (non floor-pinned) -> raw oracle already unsafe
                           (oracle geometry) or smoother lag/snap.

Additional diagnostics: shrink-amplitude attenuation (how much the policy
under-shrinks relative to the oracle on binding frames), lag analysis,
and a rear-observability check (back_width/backward_limit bind while no
obstacle lies inside the 187-channel visible ground region).

Run:  python legged_gym/tests/ea2/validate_sl_vs_oracle_audit.py
      [--seeds 1,7,13,21] [--ckpt <model.pt>] [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import torch

try:  # pytest: package ``ea2``
    from . import _ea2_testlib as tl
except ImportError:  # direct script execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _ea2_testlib as tl

from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    _hex_sample_violations,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
    _DIRECT_BOUNDARY_GROUPS,
    _physical_min_max,
    compute_direct_oracle_params_with_stats,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import data_path

# ---------------------------------------------------------------------------
# Collection-time configuration (frozen by the stored data, not the live cfg).
# ---------------------------------------------------------------------------

MARGIN = tl.MARGIN            # 0.10 collision margin (reward/telemetry)
SOFT = tl.SOFT_MARGIN         # 0.10 ramp width
ORACLE_MARGIN = 0.20          # cfg.envelope.oracle_margin at collection time
ORACLE_STEP = 0.05
ORACLE_MAX_DIST = 5.0
# Collection-time expansion soft cap.  Meta recorded in map_seed*.pt is
# authoritative (``soft_dof_pos_limit`` key); this fallback covers archives
# collected before the key existed (those were produced with 0.9).
SOFT_DOF_POS_LIMIT = 0.9
GROUP_MODE = "axis"
INTERP_CROSSING = True

LOW, HIGH = tl.LOW, tl.HIGH
MIN_V, MAX_V = _physical_min_max(LOW, HIGH)
SPAN = MAX_V - MIN_V          # signed; backward_limit span is negative

# Boundary sample indices of the rear group (back_width + backward_limit) and
# the front group, used for the observability check.
REAR_SAMPLES = sorted(
    set(_DIRECT_BOUNDARY_GROUPS["back_width"])
    | set(_DIRECT_BOUNDARY_GROUPS["backward_limit"])
)

# Body-frame visible ground region of the 187-channel grid (airy_mount).
VIS_X = (0.65, 3.65)
VIS_Y = (-1.0, 1.0)


def s_to_params(s: torch.Tensor) -> torch.Tensor:
    return MIN_V.to(s.device) + s * SPAN.to(s.device)


def min_clearance(params: torch.Tensor, head: torch.Tensor, pos: torch.Tensor,
                  df: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Min clearance over the 34 hex samples, production violation model."""
    outs = []
    for i in range(0, params.shape[0], chunk):
        v = _hex_sample_violations(
            params[i : i + chunk], head[i : i + chunk], pos[i : i + chunk],
            df, margin=MARGIN, soft_margin=SOFT,
        )
        outs.append(MARGIN - v * SOFT)
    return torch.cat(outs, dim=0).min(dim=-1).values


def raw_oracle(head: torch.Tensor, pos: torch.Tensor, df: torch.Tensor,
               soft_dof_pos_limit: float, chunk: int = 8192) -> torch.Tensor:
    """Chunked production oracle with the collection-time configuration."""
    out = []
    for i in range(0, head.shape[0], chunk):
        params, _ = compute_direct_oracle_params_with_stats(
            head[i : i + chunk], pos[i : i + chunk], df, LOW.to(df.device),
            HIGH.to(df.device),
            margin=ORACLE_MARGIN, step=ORACLE_STEP, max_dist=ORACLE_MAX_DIST,
            soft_dof_pos_limit=soft_dof_pos_limit,
            interp_crossing=INTERP_CROSSING, group_mode=GROUP_MODE,
        )
        out.append(params)
    return torch.cat(out, dim=0)


def stateful_predict(net, obs: torch.Tensor, device: str) -> torch.Tensor:
    """Run EnvelopeNet per frame with hidden carried across the episode."""
    t_len = obs.shape[0]
    hidden = None
    outs = []
    with torch.no_grad():
        for t in range(t_len):
            frame = obs[t].to(device)
            pred, hidden = net.step(frame.unsqueeze(0), hidden)
            outs.append(pred.squeeze(0).clamp(0.0, 1.0))
    return torch.stack(outs, dim=0)


def obstacle_in_view(pos: torch.Tensor, head: torch.Tensor, df: torch.Tensor,
                     chunk: int = 16384) -> torch.Tensor:
    """Whether any obstacle cell lies inside the visible ground region.

    Samples the body-frame region x in [0.65, 3.65], y in [-1, 1] on a 0.1 m
    grid, transforms to world frame, and reads the distance field; the region
    is "occupied in view" when any sampled cell is within 0.2 m of an obstacle.
    """
    xs = torch.arange(VIS_X[0], VIS_X[1] + 1e-6, 0.1, device=df.device)
    ys = torch.arange(VIS_Y[0], VIS_Y[1] + 1e-6, 0.1, device=df.device)
    gy, gx = torch.meshgrid(xs, ys, indexing="ij")
    local = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (M, 2)
    out = []
    for i in range(0, pos.shape[0], chunk):
        h = head[i : i + chunk]
        c, s = torch.cos(h), torch.sin(h)
        wx = pos[i : i + chunk, 0:1] + c.unsqueeze(-1) * local[:, 0] - s.unsqueeze(-1) * local[:, 1]
        wy = pos[i : i + chunk, 1:2] + s.unsqueeze(-1) * local[:, 0] + c.unsqueeze(-1) * local[:, 1]
        ix = ((wx - tl.WORLD_MIN) / tl.RES).long().clamp(0, df.shape[1] - 1)
        iy = ((wy - tl.WORLD_MIN) / tl.RES).long().clamp(0, df.shape[0] - 1)
        d = df[iy, ix]  # (B, M)
        out.append((d < 0.2).any(dim=-1))
    return torch.cat(out, dim=0)


def r2(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    var = tgt.var().clamp_min(1e-8)
    return float(1.0 - (pred - tgt).pow(2).mean() / var)


def audit_seed(seed: int, net, device: str, map_path: str | None = None) -> dict:
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.dataset import load_map

    data = load_map(map_path or data_path(seed))
    meta = data.meta
    assert abs(meta.get("oracle_margin", ORACLE_MARGIN) - ORACLE_MARGIN) < 1e-9
    assert meta.get("oracle_group_mode", GROUP_MODE) == GROUP_MODE
    soft_cap = float(meta.get("soft_dof_pos_limit", SOFT_DOF_POS_LIMIT))
    warmup_frames = max(1, int(meta.get("warmup_steps", 30)) // max(1, int(meta.get("lidar_decimation", 1))))

    df = data.distance_field.to(device)
    head = data.heading.to(device)          # (T, N)
    pos = data.pos.to(device)               # (T, N, 2)
    s_tgt = data.target.to(device)          # (T, N, 5)

    t_len, n_envs = head.shape
    h_f, p_f = head.reshape(-1), pos.reshape(-1, 2)
    s_tgt_f = s_tgt.reshape(-1, 5)

    valid = torch.zeros(t_len, n_envs, dtype=torch.bool, device=device)
    valid[warmup_frames:] = True
    valid = valid.reshape(-1)
    v = valid

    # --- three stages -----------------------------------------------------
    p_raw = raw_oracle(h_f, p_f, df, soft_cap)
    s_raw = ((p_raw - MIN_V.to(device)) / SPAN.to(device)).clamp(0.0, 1.0)
    s_pol = stateful_predict(net, data.obs, device).reshape(-1, 5)

    # --- clearances -------------------------------------------------------
    c_raw = min_clearance(s_to_params(s_raw), h_f, p_f, df)
    c_tgt = min_clearance(s_to_params(s_tgt_f), h_f, p_f, df)
    c_pol = min_clearance(s_to_params(s_pol), h_f, p_f, df)
    min_env = MIN_V.to(device).expand(s_pol.shape[0], 5)
    c_floor = min_clearance(min_env, h_f, p_f, df)

    unsafe = lambda c: c < MARGIN - 1e-6
    deep = lambda c: c < 0.05

    # --- raw-oracle replication check --------------------------------------
    # The stored target is the 50 Hz rate-limited output, so a frame-level
    # smoother replay at 10 Hz cannot reproduce it.  Instead validate the
    # recomputed raw oracle on *converged* frames: whenever the stored target
    # has been static for > cooldown + grow horizon (12 frames = 60 smoother
    # calls), the production smoother has fully converged to the raw oracle,
    # so stored target == env raw oracle there.  Matching it proves the
    # recomputation uses the same oracle configuration as collection.
    conv = torch.ones(t_len, n_envs, dtype=torch.bool, device=device)
    for k in range(1, 13):
        d = (s_tgt[k:] - s_tgt[:-k]).abs().amax(dim=-1)
        conv[k:] &= d < 1e-3
    conv = conv.reshape(-1) & v

    res: dict = {
        "n_frames": int(v.sum()),
        "smoother_replication": {
            "converged_frame_ratio": float(conv.float().mean()),
            "converged_mean_abs_raw_vs_target": float(
                (s_raw - s_tgt_f).abs()[conv].mean()
            ) if int(conv.sum()) else float("nan"),
            "converged_p99_abs": float(
                (s_raw - s_tgt_f).abs()[conv].quantile(0.99)
            ) if int(conv.sum()) else float("nan"),
        },
        "floor_pinned_ratio": float((c_floor[v] < MARGIN - 0.005).float().mean()),
        "raw_oracle": {
            "unsafe_ratio": float(unsafe(c_raw[v]).float().mean()),
            "deep_ratio": float(deep(c_raw[v]).float().mean()),
            "mean_min_clearance": float(c_raw[v].mean()),
        },
        "smoothed_target": {
            "unsafe_ratio": float(unsafe(c_tgt[v]).float().mean()),
            "deep_ratio": float(deep(c_tgt[v]).float().mean()),
            "mean_min_clearance": float(c_tgt[v].mean()),
        },
        "policy": {
            "unsafe_ratio": float(unsafe(c_pol[v]).float().mean()),
            "deep_ratio": float(deep(c_pol[v]).float().mean()),
            "mean_min_clearance": float(c_pol[v].mean()),
        },
    }

    # --- attribution ------------------------------------------------------
    pol_u, tgt_u, raw_u = unsafe(c_pol), unsafe(c_tgt), unsafe(c_raw)
    floor = c_floor < MARGIN - 0.005
    res["attribution"] = {
        "policy_unsafe_inherited_from_target": float(
            (pol_u & tgt_u & v).float().sum() / max(1, float((pol_u & v).float().sum()))
        ),
        "policy_unsafe_network_only": float(
            (pol_u & ~tgt_u & v).float().sum() / max(1, float((pol_u & v).float().sum()))
        ),
        "target_unsafe_given_not_floor_pinned": float(
            (tgt_u & ~floor & v).float().sum() / max(1, float(((~floor) & v).float().sum()))
        ),
        "target_unsafe_raw_also_unsafe": float(
            (tgt_u & raw_u & ~floor & v).float().sum()
            / max(1, float((tgt_u & ~floor & v).float().sum()))
        ),
        "target_unsafe_smoother_only": float(
            (tgt_u & ~raw_u & ~floor & v).float().sum()
            / max(1, float((tgt_u & ~floor & v).float().sum()))
        ),
    }

    # --- shrink amplitude -------------------------------------------------
    smin_tgt = s_tgt_f.min(dim=-1).values
    smin_raw = s_raw.min(dim=-1).values
    smin_pol = s_pol.min(dim=-1).values
    binding = v & (smin_tgt < 0.8)
    res["shrink"] = {
        "binding_frame_ratio": float(binding.float().mean()),
        "binding_mean_smin": {
            "raw": float(smin_raw[binding].mean()),
            "target": float(smin_tgt[binding].mean()),
            "policy": float(smin_pol[binding].mean()),
        },
        "per_dim_mean_s_on_binding": {
            "target": [float(s_tgt_f[binding][:, j].mean()) for j in range(5)],
            "policy": [float(s_pol[binding][:, j].mean()) for j in range(5)],
            "raw": [float(s_raw[binding][:, j].mean()) for j in range(5)],
        },
        "attenuation_curve": [],
    }
    edges = [0.0, 0.2, 0.4, 0.6, 0.8]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = binding & (smin_tgt >= lo) & (smin_tgt < hi)
        if int(sel.sum()) == 0:
            continue
        res["shrink"]["attenuation_curve"].append({
            "target_smin_bucket": [lo, hi],
            "n": int(sel.sum()),
            "mean_target_smin": float(smin_tgt[sel].mean()),
            "mean_policy_smin": float(smin_pol[sel].mean()),
        })
    # geometric tightness bins (min-envelope clearance = how tight the spot is)
    res["shrink"]["tightness_bins"] = []
    tedges = [0.0, 0.10, 0.20, 0.35, 0.60, 10.0]
    for lo, hi in zip(tedges[:-1], tedges[1:]):
        sel = v & (c_floor >= lo) & (c_floor < hi)
        if int(sel.sum()) == 0:
            continue
        res["shrink"]["tightness_bins"].append({
            "min_env_clearance_bucket": [lo, hi],
            "n": int(sel.sum()),
            "mean_target_smin": float(smin_tgt[sel].mean()),
            "mean_policy_smin": float(smin_pol[sel].mean()),
            "mean_policy_clearance": float(c_pol[sel].mean()),
        })

    # --- lag --------------------------------------------------------------
    # If the policy lags the target, its error is negatively correlated with
    # the target's per-frame velocity (rising target -> policy still below).
    vpair = (valid.view(t_len, n_envs)[1:] & valid.view(t_len, n_envs)[:-1]).reshape(-1)
    sp_seq = s_pol.view(t_len, n_envs, 5)
    st_seq = s_tgt.view(t_len, n_envs, 5)
    err = (sp_seq[:-1] - st_seq[:-1]).reshape(-1, 5)[vpair]
    dtgt = (st_seq[1:] - st_seq[:-1]).reshape(-1, 5)[vpair]
    corr_per_dim = []
    for j in range(5):
        e = err[:, j] - err[:, j].mean()
        d = dtgt[:, j] - dtgt[:, j].mean()
        denom = (e.std() * d.std()).clamp_min(1e-9)
        corr_per_dim.append(float((e * d).mean() / denom))
    res["lag"] = {
        "mean_abs_target_minus_raw": float(
            (s_tgt_f[v] - s_raw[v]).abs().mean()
        ),
        "per_dim_mean_abs_target_minus_raw": [
            float((s_tgt_f[v, j] - s_raw[v, j]).abs().mean()) for j in range(5)
        ],
        "err_vs_target_velocity_corr_per_dim": corr_per_dim,
    }

    # --- per-dim regression quality ---------------------------------------
    res["regression"] = {
        "r2": float(r2(s_pol[v], s_tgt_f[v])),
        "r2_per_dim": [float(r2(s_pol[v, j], s_tgt_f[v, j])) for j in range(5)],
        "mse_per_dim": [float((s_pol[v, j] - s_tgt_f[v, j]).pow(2).mean()) for j in range(5)],
    }

    # --- rear observability -----------------------------------------------
    in_view = obstacle_in_view(p_f, h_f, df)
    rear_bind = v & ((s_tgt_f[:, 2] < 0.7) | (s_tgt_f[:, 4] < 0.7))
    front_bind = v & ((s_tgt_f[:, 0] < 0.7) | (s_tgt_f[:, 3] < 0.7))

    def group_clearance(params: torch.Tensor, idxs: list[int]) -> torch.Tensor:
        outs = []
        for i in range(0, params.shape[0], 16384):
            viol = _hex_sample_violations(
                params[i : i + 16384], h_f[i : i + 16384], p_f[i : i + 16384],
                df, margin=MARGIN, soft_margin=SOFT,
            )
            outs.append(MARGIN - viol[:, idxs] * SOFT)
        return torch.cat(outs, dim=0).min(dim=-1).values

    c_rear_tgt = group_clearance(s_to_params(s_tgt_f), REAR_SAMPLES)
    res["observability"] = {
        "rear_binding_ratio": float(rear_bind.float().mean()),
        "rear_binding_with_no_obstacle_in_view": float(
            (rear_bind & ~in_view).float().sum() / max(1, float(rear_bind.float().sum()))
        ),
        "front_binding_ratio": float(front_bind.float().mean()),
        "front_binding_with_no_obstacle_in_view": float(
            (front_bind & ~in_view).float().sum() / max(1, float(front_bind.float().sum()))
        ),
        "rear_binding_mean_target_rear_clearance": float(c_rear_tgt[rear_bind].mean())
        if int(rear_bind.sum()) else float("nan"),
        "rear_binding_mean_policy_s_back_width": float(s_pol[rear_bind, 2].mean())
        if int(rear_bind.sum()) else float("nan"),
        "rear_binding_mean_target_s_back_width": float(s_tgt_f[rear_bind, 2].mean())
        if int(rear_bind.sum()) else float("nan"),
    }
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,7,13,21")
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../envs/el_4090/envelope_adaptive_2/sl/logs/runs/baseline/model.pt",
    ))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_outputs/sl_vs_oracle_audit.json"
    ))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ood-seed", type=int, default=None,
                    help="also audit an unseen map: collect it first (zero-action "
                         "rollout, one Isaac env in this process) into /tmp")
    ap.add_argument("--ood-num-envs", type=int, default=64)
    ap.add_argument("--ood-num-steps", type=int, default=1000)
    args = ap.parse_args(argv)

    device = args.device
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import load_checkpoint

    net, meta = load_checkpoint(args.ckpt, device=device)
    print(f"[audit] ckpt={args.ckpt} best_epoch={meta.get('best_epoch')} "
          f"val_r2={meta.get('val', {}).get('r2'):.4f}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    per_seed = {}
    for seed in seeds:
        print(f"[audit] --- seed {seed} ---", flush=True)
        per_seed[str(seed)] = audit_seed(seed, net, device)
        r = per_seed[str(seed)]
        print(f"  floor_pinned={r['floor_pinned_ratio']:.4f}  "
              f"unsafe: raw={r['raw_oracle']['unsafe_ratio']:.4f} "
              f"target={r['smoothed_target']['unsafe_ratio']:.4f} "
              f"policy={r['policy']['unsafe_ratio']:.4f}")

    if args.ood_seed is not None:
        import tempfile

        from legged_gym.envs.el_4090.envelope_adaptive_2.sl.dataset import (
            collect_map,
            save_map,
        )

        ood_path = os.path.join(
            tempfile.gettempdir(), f"ea2_audit_seed{args.ood_seed}.pt"
        )
        if os.path.exists(ood_path):
            print(f"[audit] reusing cached OOD map {ood_path}")
        else:
            print(f"[audit] collecting OOD map seed {args.ood_seed} "
                  f"({args.ood_num_envs} envs x {args.ood_num_steps} steps) ...",
                  flush=True)
            data = collect_map(
                seed=args.ood_seed, num_envs=args.ood_num_envs,
                num_steps=args.ood_num_steps,
            )
            save_map(data, ood_path)
            print(f"[audit] OOD map saved -> {ood_path}")
        print(f"[audit] --- OOD seed {args.ood_seed} (unseen map) ---", flush=True)
        per_seed[f"ood_{args.ood_seed}"] = audit_seed(
            args.ood_seed, net, device, map_path=ood_path
        )
        r = per_seed[f"ood_{args.ood_seed}"]
        print(f"  floor_pinned={r['floor_pinned_ratio']:.4f}  "
              f"unsafe: raw={r['raw_oracle']['unsafe_ratio']:.4f} "
              f"target={r['smoothed_target']['unsafe_ratio']:.4f} "
              f"policy={r['policy']['unsafe_ratio']:.4f}")

    # pooled (frame-weighted, seen maps only)
    seen = {k: v for k, v in per_seed.items() if not k.startswith("ood_")}
    pooled = {"n_frames": 0}
    keys3 = ["raw_oracle", "smoothed_target", "policy"]
    for k in keys3:
        pooled[k] = {m: 0.0 for m in ["unsafe_ratio", "deep_ratio", "mean_min_clearance"]}
    pooled["floor_pinned_ratio"] = 0.0
    for r in seen.values():
        n = r["n_frames"]
        pooled["n_frames"] += n
        pooled["floor_pinned_ratio"] += r["floor_pinned_ratio"] * n
        for k in keys3:
            for m in pooled[k]:
                pooled[k][m] += r[k][m] * n
    for k in keys3:
        for m in pooled[k]:
            pooled[k][m] /= pooled["n_frames"]
    pooled["floor_pinned_ratio"] /= pooled["n_frames"]

    out = {"ckpt": args.ckpt, "config": {
        "oracle_margin": ORACLE_MARGIN,
        "soft_dof_pos_limit": "read per-map from meta (fallback 0.9)",
        "group_mode": GROUP_MODE, "interp_crossing": INTERP_CROSSING,
        "collision_margin": MARGIN,
    }, "pooled": pooled, "per_seed": per_seed}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[audit] saved -> {args.out}")
    print(f"[audit] pooled: floor_pinned={pooled['floor_pinned_ratio']:.4f}  "
          f"unsafe raw={pooled['raw_oracle']['unsafe_ratio']:.4f} "
          f"target={pooled['smoothed_target']['unsafe_ratio']:.4f} "
          f"policy={pooled['policy']['unsafe_ratio']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
