#!/usr/bin/env python
"""One-command orchestration of the whole supervised pipeline.

Runs the atomic scripts in order, each in its **own process**, because Isaac Gym
supports only a single environment per process (a second one segfaults) and
because stages have very different costs -- re-running training should not force
re-collection.

    collect  (one subprocess per map seed)
      -> train
        -> eval   (one subprocess per map seed)
          -> export          (writes an rsl_rl-loadable checkpoint)
            -> ppo_continue  (optional; scratch vs SL-init comparison)

Examples
--------
Full run::

    python -m ...sl.scripts.pipeline --run-name baseline

Skip collection, reuse existing data, 100 PPO iterations::

    python -m ...sl.scripts.pipeline --run-name baseline --skip-collect --ppo-iters 100

Only re-evaluate and re-export::

    python -m ...sl.scripts.pipeline --run-name baseline --stages eval,export
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import (
    SLConfig,
    available_seeds,
    run_dir,
)

_PKG = "legged_gym.envs.el_4090.envelope_adaptive_2.sl.scripts"
_MODULES = {
    "collect": f"{_PKG}.collect",
    "train": f"{_PKG}.train",
    "eval": f"{_PKG}.eval",
    "export": f"{_PKG}.export",
    "ppo": f"{_PKG}.ppo_continue",
}
ALL_STAGES = ["collect", "train", "eval", "export", "ppo"]


def _run(module: str, args: List[str], stage: str) -> bool:
    cmd = [sys.executable, "-m", module, *args]
    print(f"[pipeline] --- {stage} ---", flush=True)
    print(f"[pipeline] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    status = "ok" if rc == 0 else f"FAILED (exit {rc})"
    print(f"[pipeline] {stage}: {status} in {dt:.0f}s", flush=True)
    return rc == 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-name", default="baseline")
    ap.add_argument("--stages", default=None,
                    help=f"comma separated subset of {','.join(ALL_STAGES)} (default: all)")
    ap.add_argument("--seeds", default=None,
                    help="map seeds to collect/evaluate (default: all found in data dir)")
    ap.add_argument("--pillar-counts", default=None,
                    help="obstacle counts paired with seeds ('-' = config default)")
    ap.add_argument("--skip-collect", action="store_true",
                    help="shorthand for excluding the collect stage")
    ap.add_argument("--num-envs", type=int, default=96)
    ap.add_argument("--num-steps", type=int, default=1400, help="collection steps per map")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=700)
    ap.add_argument("--ppo-iters", type=int, default=0, help="0 disables the PPO comparison")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    stages = (
        [s.strip() for s in args.stages.split(",") if s.strip()]
        if args.stages
        else list(ALL_STAGES)
    )
    if args.skip_collect and "collect" in stages:
        stages.remove("collect")
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}; valid: {ALL_STAGES}")

    out_dir = run_dir(args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "model.pt")
    print(f"[pipeline] run_name={args.run_name} stages={stages}")
    print(f"[pipeline] run dir -> {out_dir}")

    # ---- collect: one process per seed ----
    if "collect" in stages:
        cfg = SLConfig()
        if args.seeds:
            seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
            counts = [None] * len(seeds)
            if args.pillar_counts:
                parts = [p.strip() for p in args.pillar_counts.split(",")]
                counts = [None if p in ("", "-", "none") else int(p) for p in parts]
                if len(counts) == 1:
                    counts = counts * len(seeds)
        else:
            specs = cfg.data.seed_specs()
            seeds = [s for s, _ in specs]
            counts = [c for _, c in specs]

        for seed, count in zip(seeds, counts):
            ok = _run(_MODULES["collect"], [
                "--seeds", str(seed),
                "--pillar-counts", str(count) if count is not None else "-",
                "--num-envs", str(args.num_envs),
                "--num-steps", str(args.num_steps),
            ], f"collect seed={seed}")
            if not ok:
                print(f"[pipeline] aborting: collection failed for seed {seed}")
                return 1

    # ---- train ----
    if "train" in stages:
        ok = _run(_MODULES["train"], [
            "--run-name", args.run_name,
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--device", args.device,
        ], "train")
        if not ok:
            return 1

    # ---- eval ----
    if "eval" in stages:
        if not os.path.exists(ckpt):
            print(f"[pipeline] no checkpoint at {ckpt}; run the train stage first")
            return 1
        ok = _run(_MODULES["eval"], [
            "--ckpt", ckpt,
            "--num-envs", "64",
            "--num-steps", str(args.eval_steps),
            "--device", args.device,
        ], "eval")
        if not ok:
            return 1

    # ---- export ----
    if "export" in stages:
        if not os.path.exists(ckpt):
            print(f"[pipeline] no checkpoint at {ckpt}; run the train stage first")
            return 1
        ok = _run(_MODULES["export"], [
            "--ckpt", ckpt,
            "--out", os.path.join(out_dir, "policy_init.pt"),
            "--run-name", args.run_name,
        ], "export")
        if not ok:
            return 1

    # ---- ppo comparison ----
    if "ppo" in stages:
        if args.ppo_iters <= 0:
            print("[pipeline] ppo stage skipped (--ppo-iters 0)")
        else:
            seeds = available_seeds()
            cross_seed = (max(seeds) + 100) if seeds else 999
            for arm, seed in (("scratch", seeds[0] if seeds else 1),
                              ("sl_init", seeds[0] if seeds else 1),
                              ("sl_init_cross", cross_seed)):
                _run(_MODULES["ppo"], [
                    "--arm", arm,
                    "--ckpt", ckpt,
                    "--seed", str(seed),
                    "--num-envs", "64",
                    "--iterations", str(args.ppo_iters),
                    "--out", os.path.join(out_dir, f"ppo_{arm.replace('_', '')}.json"),
                    "--device", args.device,
                ], f"ppo {arm}")

    print(f"[pipeline] done. artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
