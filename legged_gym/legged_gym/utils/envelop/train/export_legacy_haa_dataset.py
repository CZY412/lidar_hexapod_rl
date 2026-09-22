"""Export bootstrap envelope rows by executing the deployed ``haa_range.pt``.

This exporter is intentionally a compatibility/data-pipeline tool.  The old
network receives only the compact symmetric condition and cannot reconstruct a
LiDAR-derived asymmetric ``allowed_support``.  It therefore writes a synthetic
symmetric support vector alongside its own range prediction.  The new geometry
trainer uses the support vector to make *new* labels; it does not copy the old
range as a target.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch

try:  # Package import when called from the simulator tree.
    from ..network.haa_swing_range import (
        HaaRangeNetwork,
        apply_env_morphology_priors,
        load_envelope_condition_spec,
    )
    from .package_haa_range import condition_footprint_support, support_directions
except ImportError:  # Direct-file execution: avoid legged_gym.utils/__init__.py.
    from _standalone import load_haa_range_module
    from package_haa_range import condition_footprint_support, support_directions
    _haa_module = load_haa_range_module()
    HaaRangeNetwork = _haa_module.HaaRangeNetwork
    apply_env_morphology_priors = _haa_module.apply_env_morphology_priors
    load_envelope_condition_spec = _haa_module.load_envelope_condition_spec


def _load_conditions(path: Path, width: int) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    rows = payload["condition"] if isinstance(payload, dict) else payload
    rows = torch.as_tensor(rows, dtype=torch.float32)
    if rows.ndim != 2 or rows.shape[1] != width:
        raise ValueError(f"--condition-dataset must contain condition [N,{width}]")
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("legged_gym/legged_gym/envs/el_4090/spider_envelop_2/envelop_network/haa_range.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--condition-dataset", type=Path, help="optional .pt tensor or {'condition': [N,8]}")
    parser.add_argument("--synthetic-body-margin", type=float, default=0.45)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    torch.manual_seed(args.seed)
    spec = load_envelope_condition_spec()
    if args.condition_dataset:
        condition = _load_conditions(args.condition_dataset, len(spec.condition_names))
        condition_source = str(args.condition_dataset)
    else:
        low, high = spec.bounds(torch.device("cpu"))
        condition = low + torch.rand(args.samples, low.numel()) * (high - low)
        condition_source = "uniform configuration distribution"
    # Priors are derived exactly as in EnvelopeConditionState, preventing an
    # old checkpoint from seeing impossible geometry/prior combinations.
    condition = apply_env_morphology_priors(condition, spec)

    device = torch.device(args.device)
    network = HaaRangeNetwork.from_checkpoint(args.checkpoint, device=device)
    if network.config.condition_names != spec.condition_names:
        raise ValueError(
            "Checkpoint condition order does not match spider_envelop condition order: "
            f"{network.config.condition_names} != {spec.condition_names}"
        )
    with torch.no_grad():
        legacy_ranges = network(condition.to(device)).cpu()
    directions = support_directions(args.directions, dtype=torch.float32)
    allowed_support = condition_footprint_support(
        condition, directions, spec.condition_names, body_margin=args.synthetic_body_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "condition": condition.cpu(),
        "allowed_support": allowed_support.cpu(),
        "legacy_haa_ranges": legacy_ranges,
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "condition_source": condition_source,
            "support_source": "symmetric legacy footprint plus synthetic_body_margin",
            "directions": args.directions,
            "synthetic_body_margin": args.synthetic_body_margin,
            "seed": args.seed,
            "haa_order": tuple(network.config.leg_names),
        },
    }, args.output)
    print(
        f"saved={args.output} rows={condition.shape[0]} "
        f"legacy_ranges={tuple(legacy_ranges.shape)} allowed_support={tuple(allowed_support.shape)}"
    )


if __name__ == "__main__":
    main()
