#!/usr/bin/env python3
"""Distil fixed-spider API HAA running ranges into a five-input network.

The label is a *safe/running* interval, never a rejected interval.  For each
leg the API's rejected intervals are complemented inside its physical URDF HAA
limits. The selected safe segment contains spider default HAA=0 when possible.
Otherwise front legs choose the negative (forward-splay) side and middle/rear
legs choose the positive (rearward-splay) side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


CONDITION_NAMES = ("front_width", "middle_width", "back_width", "forward_limit", "backward_limit")
LEG_NAMES = ("LB", "LF", "LM", "RB", "RF", "RM")
CONDITION_LOW = (0.3, 0.3, 0.3, 0.6, -0.9)
CONDITION_HIGH = (0.6, 0.7, 0.6, 0.9, -0.6)
# LB, LF, LM, RB, RF, RM. These are search bounds only, not a mammal pose.
HAA_LOW = (0.0, -1.308, 0.0, 0.0, -1.308, 0.0)
HAA_HIGH = (1.308, 0.0, 1.308, 1.308, 0.0, 1.308)


def _load_estimator_class():
    env_dir = Path(__file__).resolve().parents[3] / "envs/el_4090/spider_envelop_3"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from modal_envelope_range import NoFeasibleHaaEnvelopeError, SpiderEnvelopeHaaRangeEstimator
    return NoFeasibleHaaEnvelopeError, SpiderEnvelopeHaaRangeEstimator


class ApiRangeLabeler:
    """CPU wrapper around the exact fixed-spider envelop_3 range algorithm."""

    def __init__(self, *, directions: int, margin: float, tolerance: float, sweep_steps: int) -> None:
        no_feasible, estimator_type = _load_estimator_class()
        self.no_feasible = no_feasible
        # This is the one current spider default: all HAA/HFE/KFE are zero.
        dof_names = tuple(f"{leg}_{joint}" for leg in LEG_NAMES for joint in ("HAA", "HFE", "KFE"))
        # The API itself uses EL4090's canonical joint ordering; the estimator
        # maps this Isaac-style order by joint name.
        self.estimator = estimator_type(
            argparse.Namespace(
                directions=directions, margin=margin, tolerance=tolerance,
                sweep_steps=sweep_steps, cache_precision=6,
            ),
            condition_names=CONDITION_NAMES,
            leg_names=LEG_NAMES,
            dof_names=dof_names,
            spider_dof_pos=torch.zeros(len(dof_names)),
            haa_search_ranges=tuple(zip(HAA_LOW, HAA_HIGH)),
        )

    def label(self, condition: Tensor) -> Tensor:
        return self.estimator.compute(condition.unsqueeze(0))[0]


class ApiHaaRangeNetwork(nn.Module):
    """Five envelope inputs to six safe ``[lower, upper]`` HAA intervals."""

    def __init__(self, hidden_dims: Sequence[int] = (256, 256, 128)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        width = len(CONDITION_NAMES)
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ELU()))
            width = hidden
        self.backbone = nn.Sequential(*layers)
        self.center_head = nn.Linear(width, len(LEG_NAMES))
        self.half_head = nn.Linear(width, len(LEG_NAMES))
        self.register_buffer("condition_low", torch.tensor(CONDITION_LOW))
        self.register_buffer("condition_high", torch.tensor(CONDITION_HIGH))
        self.register_buffer("haa_low", torch.tensor(HAA_LOW))
        self.register_buffer("haa_high", torch.tensor(HAA_HIGH))

    def forward(self, condition: Tensor) -> Tensor:
        normalized = 2.0 * (condition - self.condition_low) / (self.condition_high - self.condition_low) - 1.0
        feature = self.backbone(normalized)
        center = self.haa_low + torch.sigmoid(self.center_head(feature)) * (self.haa_high - self.haa_low)
        max_half = torch.minimum(center - self.haa_low, self.haa_high - center)
        half = torch.sigmoid(self.half_head(feature)) * max_half
        return torch.stack((center - half, center + half), dim=-1)


def generate_dataset(labeler: ApiRangeLabeler, samples: int, seed: int) -> Dict[str, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    low, high = torch.tensor(CONDITION_LOW), torch.tensor(CONDITION_HIGH)
    conditions, ranges, attempts = [], [], 0
    while len(conditions) < samples:
        attempts += 1
        condition = low + torch.rand(len(CONDITION_NAMES), generator=generator) * (high - low)
        try:
            interval = labeler.label(condition)
        except labeler.no_feasible:
            continue
        conditions.append(condition)
        ranges.append(interval)
        if len(conditions) % 100 == 0 or len(conditions) == samples:
            print(f"labels: {len(conditions)}/{samples}, acceptance={len(conditions) / attempts:.1%}")
    return {"conditions": torch.stack(conditions), "ranges": torch.stack(ranges),
            "condition_names": CONDITION_NAMES, "leg_names": LEG_NAMES,
            "condition_low": low, "condition_high": high,
            "label_source": "fixed-spider API safe HAA intervals"}


def train(dataset: Dict[str, Tensor], args) -> Dict[str, object]:
    inputs, targets = dataset["conditions"].float(), dataset["ranges"].float()
    count = inputs.shape[0]
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(args.seed))
    validation_count = max(1, int(count * args.validation_fraction))
    valid_ids, train_ids = permutation[:validation_count], permutation[validation_count:]
    loader = DataLoader(TensorDataset(inputs[train_ids], targets[train_ids]), batch_size=args.batch_size, shuffle=True)
    device = torch.device(args.device)
    model = ApiHaaRangeNetwork(tuple(args.hidden_dims)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state, best_validation = None, float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_input, batch_target in loader:
            loss = torch.mean(torch.square(model(batch_input.to(device)) - batch_target.to(device)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = torch.mean(torch.square(model(inputs[valid_ids].to(device)) - targets[valid_ids].to(device))).item()
        if validation < best_validation:
            best_validation = validation
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}/{args.epochs}: validation_mse={validation:.8f}")
    return {"state_dict": best_state, "hidden_dims": tuple(args.hidden_dims),
            "condition_names": CONDITION_NAMES, "leg_names": LEG_NAMES,
            "output_names": ("lower", "upper"), "condition_low": torch.tensor(CONDITION_LOW),
            "condition_high": torch.tensor(CONDITION_HIGH), "haa_low": torch.tensor(HAA_LOW),
            "haa_high": torch.tensor(HAA_HIGH), "validation_mse": best_validation,
            "label_source": "fixed-spider API safe HAA intervals"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--dataset", type=Path, default=Path("legged_gym/utils/envelop/data/api_haa_ranges.pt"))
    parser.add_argument("--output", type=Path, default=Path("legged_gym/utils/envelop/data/api_haa_range_network.pt"))
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--sweep-steps", type=int, default=201)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(256, 256, 128))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generate_only and args.train_only:
        raise ValueError("--generate-only and --train-only are mutually exclusive")
    if args.train_only:
        dataset = torch.load(args.dataset, map_location="cpu")
    else:
        dataset = generate_dataset(ApiRangeLabeler(directions=args.directions, margin=args.margin,
                                                   tolerance=args.tolerance, sweep_steps=args.sweep_steps),
                                   args.samples, args.seed)
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, args.dataset)
        print(f"saved {dataset['conditions'].shape[0]} API labels to {args.dataset}")
    if args.generate_only:
        return
    checkpoint = train(dataset, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved network to {args.output}; validation_mse={checkpoint['validation_mse']:.8f}")


if __name__ == "__main__":
    main()
