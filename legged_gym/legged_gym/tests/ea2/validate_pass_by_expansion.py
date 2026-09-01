#!/usr/bin/env python
"""Pass-by expansion diagnostic: does the policy open the envelope too early?

Targets the reported failure mode: while passing a long obstacle the envelope
must stay laterally shrunk; once the obstacle leaves the *forward* view (it is
still beside/behind the robot), the policy starts expanding and the rear half
clips the obstacle.

Method: on the collected 50 Hz zero-action rollouts, find "pass-by events" --
periods where the smoothed oracle's ``back_width`` extent is deeply shrunk
(s < 0.5) that end with a recovery to fully open (s > 0.9).  Align all events
at the oracle recovery frame T0 and average the oracle vs policy traces in a
window around it.  If the policy trace rises before T0, it expands while the
oracle still knows the obstacle is beside the rear -- premature expansion.

Also reports, per event, the policy's rear-boundary clearance and violation
rate during the pre-recovery window.

Run: python legged_gym/tests/ea2/validate_pass_by_expansion.py [--ckpt ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import torch

try:
    from . import _ea2_testlib as tl
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _ea2_testlib as tl

from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import data_path

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_outputs")



def analyze_map(seed: int, net, device: str, half_win: int = 100):
    import validate_sl_vs_oracle_audit as A
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.dataset import load_map
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
        _hex_sample_violations,
    )

    data = load_map(data_path(seed))
    df = data.distance_field.to(device)
    head = data.heading.to(device)
    pos = data.pos.to(device)
    s_tgt = data.target.to(device)
    T, N = head.shape

    s_pol = A.stateful_predict(net, data.obs, device)  # (T, N, 5)

    # rear boundary samples of the hexagon (back_width + backward_limit group)
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
        _DIRECT_BOUNDARY_GROUPS,
    )
    rear_idx = sorted(
        set(_DIRECT_BOUNDARY_GROUPS["back_width"])
        | set(_DIRECT_BOUNDARY_GROUPS["backward_limit"])
    )

    def rear_clearance(s: torch.Tensor) -> torch.Tensor:
        """(T, N) min clearance over the rear boundary samples."""
        params = A.s_to_params(s.reshape(-1, 5))
        outs = []
        for i in range(0, params.shape[0], 16384):
            v = _hex_sample_violations(
                params[i : i + 16384], head.reshape(-1)[i : i + 16384],
                pos.reshape(-1, 2)[i : i + 16384], df,
                margin=tl.MARGIN, soft_margin=tl.SOFT_MARGIN,
            )
            outs.append(tl.MARGIN - v[:, rear_idx] * tl.SOFT_MARGIN)
        return torch.cat(outs).min(dim=-1).values.view(T, N)

    c_tgt_rear = rear_clearance(s_tgt)
    c_pol_rear = rear_clearance(s_pol)

    tgt_bw = s_tgt[:, :, 2]  # back_width extent (T, N)
    pol_bw = s_pol[:, :, 2]

    traces_tgt, traces_pol, ev_stats = [], [], []
    for n in range(N):
        tight = tgt_bw[:, n] < 0.5
        # pass-by events: tight period >= 0.4 s (20 frames) ending in recovery
        t = 0
        while t < T:
            if not bool(tight[t]):
                t += 1
                continue
            start = t
            while t < T and bool(tight[t]):
                t += 1
            end = t  # first non-tight frame
            if end - start < 20 or end >= T:
                continue
            # R4: recovery search horizon adapts to speed -- at 0.1 m/s the
            # invisible tail (~1.75 m) lasts ~17.5 s, far beyond the old fixed
            # 250-frame (5 s) window, silently dropping slow-pass events.
            seg = pos[start:end, n]
            v_mean = float(
                (seg[1:] - seg[:-1]).norm(dim=-1).mean().clamp_min(1e-6) / 0.02
            )
            search = int(min(1500, max(250, 2.0 / max(v_mean, 0.05) / 0.02)))
            # recovery: target reaches > 0.9 within the adapted horizon
            rec = None
            for k in range(end, min(end + search, T)):
                if float(tgt_bw[k, n]) > 0.9:
                    rec = k
                    break
            if rec is None or rec - half_win < 0 or rec + half_win >= T:
                continue
            traces_tgt.append(tgt_bw[rec - half_win : rec + half_win, n].cpu())
            traces_pol.append(pol_bw[rec - half_win : rec + half_win, n].cpu())
            # pre-recovery window: last 1 s before the oracle starts opening
            pre = slice(rec - 50, rec)
            ev_stats.append({
                "seed": seed, "env": n, "rec_frame": rec,
                "policy_opens_early": bool(
                    float(pol_bw[pre, n].max()) > float(tgt_bw[pre, n].min()) + 0.3
                ),
                "policy_rear_violation_during_preopen": float(
                    (c_pol_rear[pre, n] < tl.MARGIN - 1e-6).float().mean()
                ),
                "target_rear_violation_during_preopen": float(
                    (c_tgt_rear[pre, n] < tl.MARGIN - 1e-6).float().mean()
                ),
                "policy_rear_clearance_drop": float(
                    c_tgt_rear[pre, n].mean() - c_pol_rear[pre, n].mean()
                ),
            })

    if not traces_tgt:
        return None

    tt = torch.stack(traces_tgt).mean(0)
    tp = torch.stack(traces_pol).mean(0)
    return {
        "n_events": len(traces_tgt),
        "profile_offsets_s": [round((i - half_win) * 0.02, 2) for i in range(0, 2 * half_win, 10)],
        "target_profile": [round(float(x), 3) for x in tt[::10]],
        "policy_profile": [round(float(x), 3) for x in tp[::10]],
        "events": ev_stats,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,7,13,21")
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../envs/el_4090/envelope_adaptive_2/sl/logs/runs/baseline_50hz/model.pt",
    ))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "pass_by_expansion.json"))
    args = ap.parse_args(argv)

    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import load_checkpoint

    net, meta = load_checkpoint(args.ckpt, device=args.device)
    print(f"[passby] ckpt={args.ckpt} best_epoch={meta.get('best_epoch')}")

    all_events = []
    summary = {}
    for seed in (int(s) for s in args.seeds.split(",") if s.strip()):
        r = analyze_map(seed, net, args.device)
        if r is None:
            print(f"[passby] seed {seed}: no pass-by events found")
            continue
        n_early = sum(1 for e in r["events"] if e["policy_opens_early"])
        viol = sum(e["policy_rear_violation_during_preopen"] > 0 for e in r["events"])
        summary[str(seed)] = {
            "n_events": r["n_events"],
            "policy_opens_early_ratio": round(n_early / max(1, r["n_events"]), 3),
            "events_with_rear_violation_during_preopen": viol,
            "profile_offsets_s": r["profile_offsets_s"],
            "target_profile": r["target_profile"],
            "policy_profile": r["policy_profile"],
        }
        all_events.extend(r["events"])
        print(f"[passby] seed {seed}: {r['n_events']} events, "
              f"policy opens early in {n_early}/{r['n_events']}")
        print(f"    target bw @[-2,-1,-0.5,0,+1,+2]s: "
              f"{[r['target_profile'][i] for i in [0, 5, 8, 10, 15, 19]]}")
        print(f"    policy bw @[-2,-1,-0.5,0,+1,+2]s: "
              f"{[r['policy_profile'][i] for i in [0, 5, 8, 10, 15, 19]]}")
        pre_viol = sum(
            e["policy_rear_violation_during_preopen"] > 0 for e in r["events"]
        )
        print(f"    rear violation during oracle-still-shrunk window: "
              f"{pre_viol}/{r['n_events']} events")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"ckpt": args.ckpt, "summary": summary, "events": all_events}, f, indent=2)
    print(f"[passby] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
