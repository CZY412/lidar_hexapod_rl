#!/usr/bin/env python3
"""Build the offline, collision-filtered five-parameter envelope bank.

The five inputs are continuous, so "all values" means every point of a
declared regular grid.  For each point this script predicts its HAA intervals,
holds HFE/KFE at spider_envelop_3's single ``default_dof_pos`` values, checks
all 2^3 coupled HAA corners, and writes only safe parameter vectors.  The
collision margin inflates capsule radii, making the result intentionally more
conservative than the nominal proxy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from train_api_haa_range import ApiHaaRangeNetwork, CONDITION_HIGH, CONDITION_LOW, CONDITION_NAMES


LEGS = ("LB", "LF", "LM", "RB", "RF", "RM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(21,),
                        help="grid count for every parameter, or five individual counts")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--collision-margin", type=float, default=0.02,
                        help="extra capsule radius [m], in addition to nominal geometry")
    parser.add_argument("--save-rejected", action="store_true",
                        help="also materialize rejected conditions/ranges (normally use feasibility_mask instead)")
    return parser.parse_args()


def _grid(steps: tuple[int, ...]) -> torch.Tensor:
    if len(steps) == 1:
        steps *= len(CONDITION_NAMES)
    if len(steps) != len(CONDITION_NAMES) or min(steps) < 2:
        raise ValueError("--steps needs one integer >=2, or five integers >=2")
    axes = [torch.linspace(low, high, count) for low, high, count in zip(CONDITION_LOW, CONDITION_HIGH, steps)]
    return torch.cartesian_prod(*axes)


def _checker(margin: float):
    env_dir = Path(__file__).resolve().parents[3] / "envs/el_4090/spider_envelop_3"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from haa_interference import El4090HaaInterferenceChecker
    dof_names = [f"{leg}_{joint}" for leg in LEGS for joint in ("HAA", "HFE", "KFE")]
    # These values must match El4090Envelop3Cfg.init_state.default_joint_angles.
    default = torch.tensor([0.405 if name.endswith("HFE") else -0.39 if name.endswith("KFE") else 0.
                            for name in dof_names])
    return El4090HaaInterferenceChecker(dof_names, default, safety_margin=margin)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.collision_margin < 0:
        raise ValueError("--batch-size must be positive and --collision-margin must be non-negative")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if tuple(checkpoint.get("condition_names", ())) != CONDITION_NAMES:
        raise ValueError("checkpoint does not use the required five envelope inputs")
    network = ApiHaaRangeNetwork(tuple(checkpoint["hidden_dims"]))
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    steps = tuple(args.steps)
    if len(steps) == 1:
        steps *= len(CONDITION_NAMES)
    grid = _grid(steps)
    checker, safe_conditions, safe_ranges = _checker(args.collision_margin), [], []
    rejected_conditions, rejected_ranges = [], []
    # This compact table is the explicit answer to "which combinations cannot
    # be selected?"  Its index follows torch.cartesian_prod(*grid_axes).
    feasible_mask = torch.zeros(grid.shape[0], dtype=torch.bool)
    for start in range(0, grid.shape[0], args.batch_size):
        conditions = grid[start:start + args.batch_size]
        with torch.no_grad():
            ranges = network(conditions)
        valid_range = ((ranges[..., 1] - ranges[..., 0]) >= .3 - 1e-6).all(dim=1)
        safe = valid_range & checker.is_safe(ranges)
        feasible_mask[start:start + conditions.shape[0]] = safe.cpu()
        if safe.any():
            safe_conditions.append(conditions[safe])
            safe_ranges.append(ranges[safe])
        if args.save_rejected and (~safe).any():
            rejected_conditions.append(conditions[~safe])
            rejected_ranges.append(ranges[~safe])
        if start == 0 or start + args.batch_size >= grid.shape[0] or start % (args.batch_size * 100) == 0:
            print(f"checked {min(start + args.batch_size, grid.shape[0])}/{grid.shape[0]}")
    if not safe_conditions:
        raise RuntimeError("no collision-free grid points; lower --collision-margin or retrain the checkpoint")
    result = {
        "conditions": torch.cat(safe_conditions), "ranges": torch.cat(safe_ranges),
        "condition_names": CONDITION_NAMES, "grid_steps": steps,
        "grid_axes": tuple(torch.linspace(low, high, count)
                           for low, high, count in zip(CONDITION_LOW, CONDITION_HIGH, steps)),
        "feasible_mask": feasible_mask,
        "collision_margin_m": args.collision_margin,
        "hfe_kfe_default": (0.405, -0.39), "total_grid_points": grid.shape[0],
        "collision_mode": "haa_range_center_bias",
    }
    if args.save_rejected:
        result["rejected_conditions"] = torch.cat(rejected_conditions) if rejected_conditions else grid[:0]
        result["rejected_ranges"] = torch.cat(rejected_ranges) if rejected_ranges else torch.empty(0, 6, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"saved {result['conditions'].shape[0]}/{grid.shape[0]} feasible envelopes to {args.output} "
          f"(collision margin={args.collision_margin:.3f} m)")


if __name__ == "__main__":
    main()
