#!/usr/bin/env python
"""Collect EA2 supervised-learning data (B0).

Runs zero-action rollouts over one or more map seeds and stores the resulting
tensors for offline training.  Each seed gets its own file so maps can be mixed
and matched later.  Output defaults to ``sl/logs/data/``.

**Run one seed per process.**  Isaac Gym only supports a single environment per
process; constructing a second ``EL_4090_EA2`` in the same interpreter
segfaults.  ``pipeline.py`` handles this by spawning a subprocess per seed.
Passing several seeds to this script directly is only safe because it builds
them sequentially in one process -- which is precisely what crashes, so prefer
one seed per invocation.

Examples
--------
Collect the three default maps (seeds 1, 7, 13)::

    python -m legged_gym.envs.el_4090.envelope_adaptive_2.sl.scripts.collect \
        --seeds 1,7,13

Collect a dense variant to cover the measured weak spot::

    python -m ... collect --seeds 21 --pillar-counts 28
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

from legged_gym.envs.el_4090.envelope_adaptive_2.sl import dataset as ds
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import (
    SL_DATA_DIR,
    SLConfig,
    data_path,
)


def _parse_counts(raw: Optional[str], n: int) -> List[Optional[int]]:
    """Parse a comma separated list of pillar counts (``-`` means default)."""
    if not raw:
        return [None] * n
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) == 1:
        parts = parts * n
    if len(parts) != n:
        raise SystemExit(f"--pillar-counts needs 1 or {n} entries, got {len(parts)}")
    out: List[Optional[int]] = []
    for p in parts:
        out.append(None if p in ("", "-", "none", "None") else int(p))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="1,7,13", help="comma separated map seeds")
    ap.add_argument("--pillar-counts", default=None, help="comma separated obstacle counts ('-' = config default)")
    ap.add_argument("--num-envs", type=int, default=96)
    ap.add_argument("--num-steps", type=int, default=1400)
    ap.add_argument("--out-dir", default=None, help=f"default: {SL_DATA_DIR}")
    ap.add_argument(
        "--print-summary",
        action="store_true",
        help="build the windowed dataset afterwards and print its statistics",
    )
    args = ap.parse_args(argv)
    out_dir = args.out_dir or SL_DATA_DIR

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    counts = _parse_counts(args.pillar_counts, len(seeds))
    cfg = SLConfig()

    import os

    os.makedirs(out_dir, exist_ok=True)

    for seed, count in zip(seeds, counts):
        path = data_path(seed) if out_dir == SL_DATA_DIR else os.path.join(out_dir, f"map_seed{seed}.pt")
        t0 = time.time()
        data = ds.collect_map(
            seed=seed,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            lidar_decimation=cfg.data.lidar_decimation,
            pillar_count=count,
        )
        ds.save_map(data, path)
        meta = data.meta
        print(
            f"[collect] seed={seed} frames={data.obs.shape[0]} envs={data.obs.shape[1]} "
            f"margin={meta['oracle_margin']} group={meta['oracle_group_mode']} "
            f"pillar_count={count} -> {path} ({time.time() - t0:.0f}s)"
        )

    if args.print_summary:
        cfg = SLConfig()
        maps = [ds.load_map(os.path.join(out_dir, f"map_seed{s}.pt")) for s in seeds]
        dset = ds.SLDataset(cfg, maps)
        for k, v in dset.summary().items():
            print(f"[summary] {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
