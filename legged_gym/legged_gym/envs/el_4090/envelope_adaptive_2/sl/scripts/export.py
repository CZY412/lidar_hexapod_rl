#!/usr/bin/env python
"""Export a trained EnvelopeNet into an rsl_rl policy checkpoint (B4).

Examples
--------
    python -m legged_gym.envs.el_4090.envelope_adaptive_2.sl.scripts.export \
        --ckpt /tmp/ea2_sl_ckpt/model.pt \
        --out  /tmp/ea2_sl_ckpt/policy_init.pt \
        --std 0.5
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


from legged_gym.envs.el_4090.envelope_adaptive_2.sl import export as sexp
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import load_checkpoint
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import SLConfig


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="supervised checkpoint (sl/scripts/train.py output)")
    ap.add_argument("--out", required=True, help="destination ActorCriticRecurrent checkpoint")
    ap.add_argument("--std", type=float, default=None, help="exploration std (default: cfg.train.export_std)")
    ap.add_argument("--init-critic", action="store_true",
                    help="also copy the actor encoder into the critic (off by default)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-name", default=None,
                    help="also write to logs/<experiment>/<run-name>/model_0.pt so that "
                         "train.py --resume / play_ea2.py can load it")
    ap.add_argument("--iter", type=int, default=0,
                    help="iteration stamp written into the checkpoint (default 0)")
    args = ap.parse_args(argv)

    cfg = SLConfig()
    net, meta = load_checkpoint(args.ckpt, device=args.device)

    # snapshot the fresh policy so the critic can be checked for real
    fresh = sexp.build_ppo_policy(cfg, device=args.device, init_noise_std=args.std)
    reference = {k: v.detach().cpu().clone() for k, v in fresh.state_dict().items()}
    policy = sexp.export(
        net, cfg, init_noise_std=args.std, init_critic=args.init_critic,
        device=args.device, policy=fresh,
    )

    checks = sexp.verify_export(net, policy, reference_state=reference)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        print(f"[export] VERIFICATION FAILED: {failed}")
        return 1
    print(f"[export] verified: {len(checks)} checks passed (incl. critic isolation)")

    extra = {
        "source_sl_ckpt": args.ckpt,
        "source_best_epoch": meta.get("best_epoch"),
        "source_val_r2": meta.get("val", {}).get("r2"),
        "std": float(policy.std.flatten()[0]),
        "init_critic": bool(args.init_critic),
    }
    sexp.save_policy(policy, args.out, extra=extra, iteration=args.iter)
    print(f"[export] saved -> {args.out}  (std={float(policy.std.flatten()[0]):.2f})")

    if args.run_name:
        ppo_path = sexp.ppo_log_path(args.run_name, iteration=args.iter)
        sexp.save_policy(policy, ppo_path, extra=extra, iteration=args.iter)
        print(f"[export] PPO-reachable copy -> {ppo_path}")
        print(f"[export]   resume:  train.py --task=el4090_ea2 --resume "
              f"--load_run {args.run_name} --checkpoint {args.iter}")
        print(f"[export]   play:    play_ea2.py --task=el4090_ea2 "
              f"--load_run {args.run_name} --checkpoint {args.iter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
