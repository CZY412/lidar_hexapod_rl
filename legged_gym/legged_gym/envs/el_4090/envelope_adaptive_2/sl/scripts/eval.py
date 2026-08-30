#!/usr/bin/env python
"""Closed-loop evaluation of the EA2 supervised policy (B3).

Compares the deployed policy against baselines inside the real environment and
writes the comparison to JSON so that Gate G3 can assert on it.

Evaluating several seeds spawns one subprocess per seed: Isaac Gym only supports
a single environment per process, and building a second one after tearing the
first down segfaults.

Examples
--------
    python -m legged_gym.envs.el_4090.envelope_adaptive_2.sl.scripts.eval \
        --ckpt sl/logs/runs/baseline/model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional

#: Set in child processes so a spawned eval does not recurse.
_CHILD_ENV = "EA2_SL_EVAL_CHILD"
_MODULE = "legged_gym.envs.el_4090.envelope_adaptive_2.sl.scripts.eval"


def _spawn_per_seed(args, seeds: List[int]) -> dict:
    """Run each seed in its own process and merge the per-seed JSON files."""
    out_dir = os.path.dirname(os.path.abspath(args.ckpt)) or "."
    env = os.environ.copy()
    env[_CHILD_ENV] = "1"
    results = {}
    for seed in seeds:
        per_seed = os.path.join(out_dir, f"closed_loop_s{seed}.json")
        cmd = [
            sys.executable, "-m", _MODULE,
            "--ckpt", args.ckpt,
            "--seed", str(seed),
            "--num-envs", str(args.num_envs),
            "--num-steps", str(args.num_steps),
            "--modes", args.modes,
            "--device", args.device,
            "--out", per_seed,
        ]
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            print(f"[eval] WARNING: seed {seed} exited with {rc}, skipping")
            continue
        with open(per_seed) as f:
            results[str(seed)] = json.load(f)["results"][str(seed)]
    return results

from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import (
    evaluate_closed_loop,
    load_checkpoint,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import (
    SL_DATA_DIR,
    SLConfig,
    available_seeds,
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", default=None,
                    help="map seed(s) to evaluate on, comma separated (default: all collected)")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=700)
    ap.add_argument("--modes", default="stateful,window",
                    help="'stateful' (recommended) and/or 'window' (conservative fallback)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="default: <run-dir>/closed_loop.json (per-seed files alongside)")
    args = ap.parse_args(argv)

    cfg = SLConfig()
    net, meta = load_checkpoint(args.ckpt, device=args.device)
    print(f"[eval] ckpt={args.ckpt} best_epoch={meta.get('best_epoch')} "
          f"val_r2={meta.get('val', {}).get('r2'):.4f}")

    seeds = (
        [int(s) for s in args.seed.split(",") if s.strip()]
        if args.seed
        else available_seeds()
    )
    if not seeds:
        raise SystemExit(f"no collected maps found in {SL_DATA_DIR}; run scripts.collect first")

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    run_dir_path = os.path.dirname(os.path.abspath(args.ckpt)) or "."

    if len(seeds) > 1 and os.environ.get(_CHILD_ENV) != "1":
        # one map per process -- a second env in the same process segfaults
        print(f"[eval] evaluating {len(seeds)} seeds in separate processes: {seeds}")
        results = _spawn_per_seed(args, seeds)
        for seed, res in results.items():
            print(f"[eval] --- map seed={seed} ---")
            for name, r in res.items():
                print(f"[eval]   {name:<22}{r['step_reward']:>+10.4f}{r['pred_target_mse']:>10.5f}"
                      f"{r['collision_rate']:>10.4f}{r['temporal_std']:>9.4f}")
    else:
        if len(seeds) > 1:
            raise SystemExit("child process must receive exactly one seed")
        seed = seeds[0]
        results = {str(seed): evaluate_closed_loop(
            cfg, net, seed=seed, num_envs=args.num_envs,
            num_steps=args.num_steps, device=args.device, modes=modes,
        )}
        print(f"[eval] --- map seed={seed} ---")
        for name, r in results[str(seed)].items():
            print(f"[eval]   {name:<22}{r['step_reward']:>+10.4f}{r['pred_target_mse']:>10.5f}"
                  f"{r['collision_rate']:>10.4f}{r['temporal_std']:>9.4f}"
                  f"{r.get('area', float('nan')):>9.4f}")

    out = args.out or os.path.join(run_dir_path, "closed_loop.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"ckpt": args.ckpt, "seeds": seeds, "num_steps": args.num_steps, "modes": list(modes),
             "meta": {k: v for k, v in meta.items() if k != "state_dict"},
             "results": {str(k): v for k, v in results.items()}},
            f,
            indent=2,
        )
    print(f"[eval] saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
