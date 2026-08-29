#!/usr/bin/env python
"""Real-environment safety/functional test for the EA2 oracle pipeline.

Runs the ACTUAL Isaac Gym env (real generated pillar map, real A* paths,
50 Hz kinematics, soft replans, timeout reset) with a *perfect-oracle
policy*: each step the raw action is the exact linear-map inverse of the
oracle target computed from the env's own state, so ``actions_mapped``
tracks the oracle as well as the 50 Hz loop allows.

Verified, over one full timeout episode (1600 steps, 64 envs):

1. mechanics   -- obs finite and within [0, 1], LiDAR refresh, timeout path,
                  reward terms finite;
2. safety      -- feasible-frame worst boundary+interior sample clearance of
                  the *executed* envelope (independent check via
                  ``_hex_sample_violations``), plus the lag attribution:
                  executed-envelope violations are expected to be pure
                  one-step target lag (the fresh same-pose oracle stays
                  safe) interacting with cell-quantised EDT reads;
3. consistency -- reward-side oracle MSE of the perfect policy (inherent
                  moving-target lag), envelope_limits penalty rate (targets
                  below the soft floor after the safety-first clamp fix),
                  potential, action rate;
4. interp      -- interp on/off parameter difference on the real map;
5. observability -- oracle computed on a distance field rebuilt from
                  frustum-masked occupancy vs the true field: per-parameter
                  gap = the geometry a perception-only policy can never see.

Usage (from legged_gym/legged_gym):
    python tests/ea2/validate_ea2_env_oracle_pipeline.py \
        --task=el4090_ea2 --headless --num_envs 64
"""

from __future__ import annotations

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import sys

import numpy as np
import torch
from scipy import ndimage

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    default_params,
    envelope_action_scale,
    normalized_envelope_params,
)
from legged_gym.utils import get_args, task_registry

try:  # pytest: package ``ea2``
    from . import _ea2_testlib as tl
except ImportError:  # direct script execution
    import _ea2_testlib as tl

_MARGIN = tl.MARGIN
# 187-channel footprint (body frame), README 2.3.2
_FRUSTUM = (0.40, 3.70, -1.10, 1.10)
_N_STEPS = 1600


def _oracle(head, pos, dft, low, high, *, interp: bool):
    return tl.oracle_batch(head, pos, dft, low, high, interp=interp)


def main() -> None:
    sys.argv = [
        "validate_ea2_env_oracle_pipeline.py",
        "--task=el4090_ea2",
        "--headless",
        "--num_envs",
        "64",
    ]
    args = get_args()
    env, _ = task_registry.make_env("el4090_ea2", args)
    device = env.device
    low = env._envelope_low_dev
    high = env._envelope_high_dev
    default = default_params(low, high)
    scale = envelope_action_scale(
        low, high,
        float(env.cfg.envelope.soft_dof_pos_limit),
        float(env.cfg.envelope.action_max),
    )
    interp = bool(getattr(env.cfg.envelope, "oracle_interp_crossing", True))
    print(f"env ready: {env.num_envs} envs, device={device}, interp={interp}")

    obs, _ = env.reset()
    log = {k: [] for k in (
        "rew", "potential", "oracle_mse", "limits", "act_rate",
        "min_clear_exec", "fresh_min", "feasible", "time_out",
    )}
    poses = []
    obs_bad = 0

    dft = env.distance_field
    for t in range(_N_STEPS):
        with torch.no_grad():
            oracle = _oracle(env.heading, env.base_pos[:, :2], dft, low, high, interp=interp)
            raw = (oracle - default) / scale
            obs, _, rew, dones, infos = env.step(raw)

        # obs sanity: 187 range channels must live in [0, 1]
        if not bool(torch.isfinite(obs).all()):
            obs_bad += 1
        rng = obs[:, :187]
        if float(rng.min()) < -1e-4 or float(rng.max()) > 1.0 + 1e-4:
            obs_bad += 1

        # independent safety check of the executed envelope at the NEW pose
        head = env.heading.clone()
        pos = env.base_pos[:, :2].clone()
        poses.append((head.cpu(), pos.cpu()))
        seq = env.actions_mapped.clone()
        clr = tl.sample_clearances(seq, head, pos, dft)
        min_frame = clr.min(dim=-1).values
        feasible = tl.feasible_mask(seq, head, pos, dft, low, high)

        # fresh oracle at the SAME pose as the executed envelope: the
        # executed params are the previous-pose target, so exec-vs-fresh
        # isolates pure one-step target lag
        fresh = _oracle(head, pos, dft, low, high, interp=interp)
        log["fresh_min"].append(
            tl.sample_clearances(fresh, head, pos, dft)[:, :24].min(-1).values.cpu()
        )
        log["rew"].append(float(rew.mean()))
        log["min_clear_exec"].append(min_frame.cpu())
        log["feasible"].append(feasible.cpu())
        log["time_out"].append(bool(infos["time_outs"].any()))
        log["potential"].append(float(tl.potential(seq).mean()))
        # reward-side oracle MSE (mapped vs fresh oracle at the same pose)
        norm_a = normalized_envelope_params(seq, low, high)
        norm_o = normalized_envelope_params(oracle, low, high)
        log["oracle_mse"].append(float(((norm_a - norm_o) ** 2).mean(dim=-1).mean()))
        tgt = env.actions_target
        soft_m = (1.0 - float(env.cfg.envelope.soft_dof_pos_limit)) * (high - low) / 2
        over = ((tgt < low + soft_m) | (tgt > high - soft_m)).any(dim=-1)
        log["limits"].append(float(over.float().mean()))
        log["act_rate"].append(
            float((env.actions - env.last_actions_raw).norm(dim=-1).mean())
            if t > 0 else 0.0
        )

    feas = torch.cat(log["feasible"])
    clr = torch.cat(log["min_clear_exec"])
    print("\n" + "=" * 76)
    print(f"ROLLOUT: {_N_STEPS} steps x {env.num_envs} envs (interp={interp})")
    print("=" * 76)
    print(f"  obs violations (non-finite / out of [0,1]): {obs_bad}")
    print(f"  timeouts observed: {sum(log['time_out'])}")
    print(f"  mean reward: {np.mean(log['rew']):.4f}")
    print(f"  feasible frames: {feas.float().mean() * 100:.2f}%")
    print(f"  executed-envelope min clearance on feasible frames: "
          f"{clr[feas].min().item():.4f} m  (margin={_MARGIN})")
    viol_frames = (clr < _MARGIN - 1e-3) & feas
    n_viol = int(viol_frames.sum())
    print(f"  executed-envelope violating env-frames (feasible only): "
          f"{viol_frames.float().mean() * 100:.3f}%")

    # ---- lag attribution: fresh oracle at the same pose vs executed ----
    # Executed params are the PREVIOUS-pose oracle target; violations are
    # expected to be pure one-step target lag (fresh target stays safe)
    # interacting with cell-quantised EDT reads.
    heads = torch.cat([h for h, _ in poses])
    poss = torch.cat([p for _, p in poses])
    fresh_min = torch.cat(log["fresh_min"])
    n_fresh_safe = int((viol_frames & (fresh_min >= _MARGIN - 1e-3)).sum())
    n_fresh_viol = int((viol_frames & (fresh_min < _MARGIN - 1e-3)).sum())
    print(f"  lag attribution on {n_viol} violating env-frames: fresh target safe "
          f"{n_fresh_safe} ({100 * n_fresh_safe / max(n_viol, 1):.1f}%), "
          f"fresh also violates {n_fresh_viol}")

    print(f"  perfect-policy oracle MSE: mean={np.mean(log['oracle_mse']):.6f} "
          f"p99={np.percentile(log['oracle_mse'], 99):.6f}")
    print(f"  envelope_limits active (target outside soft range): "
          f"{np.mean(log['limits']) * 100:.2f}% of env-steps")
    print(f"  mean potential: {np.mean(log['potential']):.4f}")
    print(f"  mean raw-action L2 change: {np.mean(log['act_rate']):.4f}")

    # ---- interp on/off on the real map (subsampled frames) ----
    idx10 = torch.arange(0, _N_STEPS, 10) * env.num_envs
    o_on = _oracle(heads[idx10].to(device), poss[idx10].to(device), dft, low, high, interp=True)
    o_off = _oracle(heads[idx10].to(device), poss[idx10].to(device), dft, low, high, interp=False)
    diff = (o_on - o_off).abs()
    print("\n" + "=" * 76)
    print(f"INTERP ON/OFF on real map ({len(idx10)} frames x {env.num_envs} envs)")
    print("=" * 76)
    print("  max |diff| per param:", [round(v, 4) for v in diff.max(dim=0).values.cpu().tolist()])
    print("  mean |diff| per param:", [round(v, 4) for v in diff.mean(dim=0).cpu().tolist()])
    c_on = tl.sample_clearances(
        o_on, heads[idx10].to(device), poss[idx10].to(device), dft
    ).min(-1).values
    c_off = tl.sample_clearances(
        o_off, heads[idx10].to(device), poss[idx10].to(device), dft
    ).min(-1).values
    print(f"  worst clearance interp=True: {c_on.min().item():.4f}  False: {c_off.min().item():.4f}")

    # ---- observability gap: frustum-masked occupancy field vs true field ----
    print("\n" + "=" * 76)
    print("OBSERVABILITY GAP (oracle on frustum-masked field vs true field)")
    print("=" * 76)
    d_np = dft.cpu().numpy()
    x0, x1, y0, y1 = _FRUSTUM
    n_sample = 40
    frame_ids = torch.linspace(0, _N_STEPS - 1, n_sample).long()
    gaps = []
    for fi in frame_ids.tolist():
        h0 = heads[fi * env.num_envs : (fi + 1) * env.num_envs].to(device)
        p0 = poss[fi * env.num_envs : (fi + 1) * env.num_envs].to(device)
        true_o = _oracle(h0, p0, dft, low, high, interp=interp)
        masked_list = []
        for e in range(env.num_envs):
            cx, cy, hh = float(p0[e, 0]), float(p0[e, 1]), float(h0[e])
            cx_i, cy_i = int(round((cx + 37) / 0.1)), int(round((cy + 37) / 0.1))
            r = 60
            ys0, ys1 = max(cy_i - r, 0), min(cy_i + r + 1, 740)
            xs0, xs1 = max(cx_i - r, 0), min(cx_i + r + 1, 740)
            sl = np.s_[ys0:ys1, xs0:xs1]
            occ = d_np[sl] <= 1e-6  # occupied cells only; keep the true footprint
            ys, xs = np.mgrid[sl[0].start:sl[0].stop, sl[1].start:sl[1].stop]
            wx = -37.0 + xs * 0.1
            wy = -37.0 + ys * 0.1
            dx, dy = wx - cx, wy - cy
            bx = np.cos(hh) * dx + np.sin(hh) * dy
            by = -np.sin(hh) * dx + np.cos(hh) * dy
            in_frustum = (bx >= x0) & (bx <= x1) & (by >= y0) & (by <= y1)
            # erase unseen obstacles and REBUILD the distance field from the
            # masked occupancy (a halo of stale small-EDT cells would
            # otherwise make the unseen geometry partly visible)
            masked_occ = occ & in_frustum
            crop = ndimage.distance_transform_edt(
                ~masked_occ, sampling=(0.1, 0.1)
            ).astype(np.float32)
            full = torch.full_like(dft, 5.0)
            full[sl] = torch.as_tensor(crop, dtype=torch.float32, device=device)
            o_m = _oracle(h0[e : e + 1], p0[e : e + 1], full, low, high, interp=interp)
            masked_list.append(o_m[0])
        masked_o = torch.stack(masked_list, dim=0)
        gaps.append(
            (
                normalized_envelope_params(true_o, low, high)
                - normalized_envelope_params(masked_o, low, high)
            ).abs().cpu()
        )
    gaps = torch.cat(gaps, dim=0)
    names = ("front_width", "middle_width", "back_width", "forward_limit", "backward_limit")
    print(f"  {len(gaps)} env-frames sampled; mean |gap| in normalised extent:")
    for j, n in enumerate(names):
        print(f"    {n:15s} mean={gaps[:, j].mean().item():.4f}  p90={gaps[:, j].quantile(0.9).item():.4f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
