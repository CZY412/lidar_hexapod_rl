#!/usr/bin/env python
"""Train the EA2 supervised policy (B2).

Reads collected maps from ``sl/logs/data/`` by default and writes the run to
``sl/logs/runs/<run-name>/``: the checkpoint, its metrics, and a snapshot of the
configuration used (so a run can be reproduced later).

Examples
--------
    python -m legged_gym.envs.el_4090.envelope_adaptive_2.sl.scripts.train \
        --seeds 1,7,13 --epochs 50 --run-name baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from legged_gym.envs.el_4090.envelope_adaptive_2.sl import dataset as ds
from legged_gym.envs.el_4090.envelope_adaptive_2.sl import train as slt
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import (
    SL_DATA_DIR,
    SLConfig,
    run_dir,
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help=f"default: {SL_DATA_DIR}")
    ap.add_argument("--seeds", default=None, help="subset of available map seeds (default: all)")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="override cfg.train.seq_len (default: config value, 200 = 4 s at 50 Hz)")
    ap.add_argument("--window-stride", type=int, default=None,
                    help="override cfg.train.window_stride (default: config value, 10 = 0.2 s)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux-ks", default=None,
                    help="comma separated recall-probe horizons, e.g. 25,50,100,200,300 "
                         "(default: config value; empty disables the heads)")
    ap.add_argument("--run-name", default="baseline", help="sub-directory of sl/logs/runs/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many epochs with no val improvement (0 = off)")
    args = ap.parse_args(argv)

    data_dir = args.data_dir or SL_DATA_DIR
    out_dir = run_dir(args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    cfg = SLConfig()
    if args.seq_len is not None:
        cfg.train.seq_len = args.seq_len
    if args.window_stride is not None:
        cfg.train.window_stride = args.window_stride
    if args.aux_ks is not None:
        cfg.model.aux_ks = [int(k) for k in args.aux_ks.split(",") if k.strip()]
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr

    all_files = sorted(
        f for f in os.listdir(data_dir) if f.startswith("map_seed") and f.endswith(".pt")
    )
    if not all_files:
        raise SystemExit(f"no map_seed*.pt found in {data_dir}")
    if args.seeds:
        wanted = {int(s) for s in args.seeds.split(",")}
        all_files = [f for f in all_files if int(f[len("map_seed") : -len(".pt")]) in wanted]
        if not all_files:
            raise SystemExit(f"none of the requested seeds found in {data_dir}")

    maps = [ds.load_map(os.path.join(data_dir, f)) for f in all_files]
    print(f"[train] maps={all_files}")

    dset = ds.SLDataset(cfg, maps)
    summary = dset.summary()
    print(f"[train] windows={summary['n_windows']} train={summary['n_train']} val={summary['n_val']}")
    print(f"[train] target_std={summary['target_std']}")
    print(f"[train] corr fw~fl={summary['corr_fw_fl']:.4f}  bw~bl={summary['corr_bw_bl']:.4f}")

    result = slt.train(
        cfg, dset, device=args.device, log_every=args.log_every, patience=args.patience
    )

    print(f"[train] best_epoch={result['best_epoch']}  best_val_r2={result['best_val_r2']:.4f}")
    v = result["val"]
    print(f"[val]   r2={v['r2']:.4f}  mse={v['mse']:.5f}")
    print(f"[val]   r2_per_dim={ {k: round(x, 3) for k, x in v['r2_per_dim'].items()} }")
    print(f"[val]   mse_per_dim={ {k: round(x, 5) for k, x in v['mse_per_dim'].items()} }")
    print(f"[val]   pc_var={[round(x, 3) for x in v['pc_var']]}")
    print(f"[val]   pc_r2 ={[round(x, 3) for x in v['pc_r2']]}")
    print(f"[val]   subspace_r2={ {k: round(x, 3) for k, x in v['subspace_r2'].items()} }")
    print(
        f"[val]   temporal_std={v['temporal_std']:.4f} "
        f"(oracle {v['temporal_std_oracle']:.4f}, "
        f"ratio {v['temporal_std'] / max(v['temporal_std_oracle'], 1e-8):.3f})"
    )
    gap = result["train"]["r2"] - v["r2"]
    print(f"[val]   train_r2={result['train']['r2']:.4f}  gap={gap:.4f}")

    ckpt = os.path.join(out_dir, "model.pt")
    slt.save_checkpoint(result, cfg, ckpt)

    # snapshot the configuration so the run can be reproduced
    with open(os.path.join(out_dir, "sl_config.json"), "w") as f:
        json.dump(
            {
                "run_name": args.run_name,
                "seeds": [int(f[len("map_seed") : -len(".pt")]) for f in all_files],
                "data_dir": data_dir,
                "config": cfg.to_dict(),
            },
            f,
            indent=2,
        )
    print(f"[train] saved -> {ckpt}")
    print(f"[train] run dir -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
