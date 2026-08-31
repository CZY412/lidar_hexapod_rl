#!/usr/bin/env python
"""Stop-beside-obstacle forgetting diagnostic (deployment safety probe).

Freezes envs simultaneously while they stand beside an obstacle (oracle
back_width deeply shrunk, rear clearance currently safe) and holds them
stationary, well past the GRU's trained memory horizon.  Reports whether the
policy envelope slowly opens (forgetting of the out-of-view obstacle) and how
many frozen envs end up violating.

Also supports ``--in-view``: the control variant freezing envs with the
obstacle ahead and fully visible -- stationary there must NOT open, separating
"memory decay" from a generic stopped-state artifact.

Run: python legged_gym/tests/ea2/validate_stop_forgetting.py [--ckpt ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import torch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_outputs")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../envs/el_4090/envelope_adaptive_2/sl/logs/runs/v2_multik/model.pt",
    ))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--hold-steps", type=int, default=1800,
                    help="stationary observation steps (1800 = 36 s, < episode cap)")
    ap.add_argument("--in-view", action="store_true",
                    help="freeze with the obstacle ahead/visible instead of beside")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import validate_sl_vs_oracle_audit as A
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.dataset import build_env
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.export import build_ppo_policy
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import (
        ACTION_SIGN,
        SLConfig,
        env_action_scale,
    )
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
        _hex_sample_violations,
    )
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
        _DIRECT_BOUNDARY_GROUPS,
    )

    dev = args.device
    pol = build_ppo_policy(SLConfig(), device=dev)
    raw = torch.load(args.ckpt, map_location="cpu")
    pol.load_state_dict(raw["model_state_dict"])
    pol.to(dev).eval()

    if args.in_view:
        grp = sorted(set(_DIRECT_BOUNDARY_GROUPS["front_width"])
                     | set(_DIRECT_BOUNDARY_GROUPS["forward_limit"]))
    else:
        grp = sorted(set(_DIRECT_BOUNDARY_GROUPS["back_width"])
                     | set(_DIRECT_BOUNDARY_GROUPS["backward_limit"]))

    def group_clearance(params: torch.Tensor) -> torch.Tensor:
        viol = _hex_sample_violations(
            params, env.heading, env.base_pos[:, :2], env.distance_field,
            margin=0.10, soft_margin=0.10,
        )
        return (0.10 - viol[:, grp] * 0.10).min(dim=-1).values

    env = build_env(args.seed, args.num_envs)
    try:
        k = env_action_scale()
        sign = torch.tensor(ACTION_SIGN, device=dev)
        env.reset(); pol.reset()
        obs = env.obs_buf.clone()
        frozen = None
        with torch.inference_mode():
            for t in range(1, 2200):
                a = pol.act_inference(obs)
                out = env.step(a); obs = out[0]
                if t % 25 == 0:
                    s_t = env._oracle_smoother.prev_s
                    clr = group_clearance(A.s_to_params(0.5 + k * sign * a))
                    if args.in_view:
                        cand = ((s_t[:, 0] < 0.55) & (s_t[:, 2] > 0.9)
                                & (~env._turn_in_place) & (env.v > 0.5) & (clr >= 0.095))
                    else:
                        cand = ((s_t[:, 2] < 0.55) & (~env._turn_in_place)
                                & (env.v > 0.5) & (clr >= 0.095))
                    ids = cand.nonzero(as_tuple=False).flatten().tolist()
                    if len(ids) >= 12:
                        frozen = ids[:12]
                        for i in frozen:
                            env.v[i] = 0.0
                        # the stage-2 speed scheduler redraws v for ALL envs
                        # every speed_resample_steps -- it would un-freeze the
                        # frozen group within <=3 s.  Freeze means frozen.
                        env.cfg.path.speed_randomize = False
                        break
        assert frozen, "no simultaneous freeze point found in 2200 steps"
        ft = torch.tensor(frozen, device=dev)
        tag = "in_view" if args.in_view else "beside"
        print(f"[stop] freeze at {tag}: {frozen}")

        rec = []
        with torch.inference_mode():
            for t in range(1, args.hold_steps + 1):
                env.v[ft] = 0.0   # belt and braces: frozen stays frozen
                a = pol.act_inference(obs)
                out = env.step(a); obs = out[0]
                if t % 50 == 0:
                    s_pol = 0.5 + k * sign * a
                    clr = group_clearance(A.s_to_params(s_pol))
                    row = {
                        "t_s": round(t * 0.02, 1),
                        "bw_mean": float(s_pol[ft, 2].mean()),
                        "fw_mean": float(s_pol[ft, 0].mean()),
                        "smin_mean": float(s_pol[ft].min(dim=-1).values.mean()),
                        "clr_mean": float(clr[ft].mean()),
                        "clr_min": float(clr[ft].min()),
                        "collisions": int((env._collision_hard[ft] > 0).sum()),
                        "oracle_bw": float(env._oracle_smoother.prev_s[ft, 2].mean()),
                        "oracle_fw": float(env._oracle_smoother.prev_s[ft, 0].mean()),
                    }
                    rec.append(row)
                    print(row, flush=True)

        out_path = args.out or os.path.join(
            OUT_DIR, f"stop_forgetting_{tag}_seed{args.seed}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"ckpt": args.ckpt, "mode": tag, "frozen": frozen,
                       "hold_steps": args.hold_steps, "records": rec}, f, indent=2)
        print(f"[stop] saved -> {out_path}")
        return 0
    finally:
        del env
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
