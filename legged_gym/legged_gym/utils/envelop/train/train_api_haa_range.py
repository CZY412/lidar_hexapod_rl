#!/usr/bin/env python3
"""Distil fixed-spider API HAA running ranges into a five-input network.

The label is a *safe/running* interval, never a rejected interval.  For each
leg the API's rejected intervals are complemented inside its physical URDF HAA
limits.  It is restricted to [-1.5, 0] for front legs (forward) and [0, 1.5]
for middle/rear legs (backward).  A sample is ``none`` when any coupled running
interval is narrower than 0.2 rad.
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
# LB, LF, LM, RB, RF, RM.  These are directional running domains, not a pose.
HAA_LOW = (0.0, -1.5, 0.0, 0.0, -1.5, 0.0)
HAA_HIGH = (1.5, 0.0, 1.5, 1.5, 0.0, 1.5)
MIN_RUNNING_WIDTH = 0.2
LABEL_SOURCE = "fixed-spider API directional rejection-complement running intervals"


class NoFeasibleHaaEnvelopeError(RuntimeError):
    """The five-parameter envelope has no usable HAA interval."""


def select_largest_running_segment(
    segments: Sequence[tuple[float, float]],
    *,
    minimum_width: float = MIN_RUNNING_WIDTH,
) -> tuple[float, float]:
    """Return one contiguous running component, or report ``none``.

    Rejection can split a directional domain into several components.  They
    must not be joined because the gap is rejected; the largest component is
    the range exported to the controller.
    """
    candidates = [segment for segment in segments if segment[1] - segment[0] >= minimum_width]
    if not candidates:
        raise NoFeasibleHaaEnvelopeError(
            f"none: no >= {minimum_width:.1f} rad contiguous running interval"
        )
    return max(candidates, key=lambda segment: (segment[1] - segment[0], -abs(0.5 * (segment[0] + segment[1]))))


class ApiRangeLabeler:
    """Exact five-parameter envelope -> safe paired HAA intervals.

    This deliberately calls the maintained ``el4090_envelope`` API rather
    than a task-local copy.  HFE/KFE are pinned to the one spider default
    posture used by envelop_3; only HAA is swept.
    """

    def __init__(self, *, directions: int, margin: float, tolerance: float, sweep_steps: int) -> None:
        self.no_feasible = NoFeasibleHaaEnvelopeError  # compatibility for visualization callers
        source = Path(__file__).resolve().parents[1] / "el4090_envelope" / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from el4090_envelope import (BatchedUrdfKinematics, default_el4090_capsules,
                                     haa_rejection_ranges, load_urdf_joints,
                                     support_directions)
        urdf = Path(__file__).resolve().parents[4] / "resources/robots/el_4090/urdf/el_4090.urdf"
        self.kinematics = BatchedUrdfKinematics(load_urdf_joints(urdf))
        self.capsules = default_el4090_capsules()
        self.haa_rejection_ranges = haa_rejection_ranges
        self.directions = support_directions(directions)
        self.margin, self.tolerance, self.sweep_steps = margin, tolerance, sweep_steps
        self.lower, self.upper = self.kinematics.joint_limits(soft_fraction=1.0)
        self.haa_indices = tuple(self.kinematics.actuated_joint_names.index(f"{leg}_HAA") for leg in LEG_NAMES)
        self.pinned = torch.zeros(self.kinematics.num_dof)
        # Must agree with El4090Envelop3Cfg.init_state.default_joint_angles.
        for leg in LEG_NAMES:
            self.pinned[self.kinematics.actuated_joint_names.index(f"{leg}_HFE")] = 0.3020
            self.pinned[self.kinematics.actuated_joint_names.index(f"{leg}_KFE")] = -0.2708

    def _allowed_support(self, condition: Tensor) -> Tensor:
        front, middle, back, forward, backward = condition.tolist()
        vertices = torch.tensor(((forward, front), (0., middle), (backward, back),
                                 (backward, -back), (0., -middle), (forward, -front)))
        return (vertices @ self.directions.T).amax(dim=0) + self.margin

    @staticmethod
    def _safe_segments(low: float, high: float, rejected) -> list[tuple[float, float]]:
        result, cursor = [], low
        for left, right in sorted(rejected):
            # A rejected band may lie wholly outside the deliberately smaller
            # spider training segment (the API itself uses URDF limits).
            if right <= low or left >= high:
                continue
            left, right = max(left, low), min(right, high)
            if left > cursor:
                result.append((cursor, left))
            cursor = max(cursor, right)
        if cursor < high:
            result.append((cursor, high))
        return result

    def label(self, condition: Tensor) -> Tensor:
        rejected = self.haa_rejection_ranges(
            self.kinematics, self.capsules, self.directions,
            self._allowed_support(condition), self.lower, self.upper, self.pinned,
            self.haa_indices, tolerance=self.tolerance, sweep_steps=self.sweep_steps,
        ).per_haa_joint_intervals
        ranges = []
        for index, joint_index in enumerate(self.haa_indices):
            choices = self._safe_segments(HAA_LOW[index], HAA_HIGH[index], rejected[joint_index])
            # A rejection set can split the directional domain.  A controller
            # needs one contiguous interval, so retain the largest feasible
            # segment rather than joining separated safe components.
            try:
                ranges.append(select_largest_running_segment(choices))
            except NoFeasibleHaaEnvelopeError as error:
                raise NoFeasibleHaaEnvelopeError(f"{error} for {LEG_NAMES[index]}") from error
        output = torch.tensor(ranges, dtype=torch.float32)
        # Coupled left/right legs need the intersection, not an average: this
        # preserves safety for both individually computed API intervals.
        for left, right in ((0, 3), (1, 4), (2, 5)):
            lo, hi = max(float(output[left, 0]), float(output[right, 0])), min(float(output[left, 1]), float(output[right, 1]))
            if hi - lo < MIN_RUNNING_WIDTH:
                raise NoFeasibleHaaEnvelopeError(
                    f"none: paired {LEG_NAMES[left]}/{LEG_NAMES[right]} running interval is too narrow"
                )
            output[left] = output[right] = torch.tensor((lo, hi))
        return output


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
        # Guarantee every prediction is suitable for HaaRangeState before the
        # collision filter sees it: lower < upper and width >= 0.2 rad.
        minimum_half = 0.1
        center = self.haa_low + minimum_half + torch.sigmoid(self.center_head(feature)) * (self.haa_high - self.haa_low - 2.0 * minimum_half)
        max_half = torch.minimum(center - self.haa_low, self.haa_high - center)
        half = minimum_half + torch.sigmoid(self.half_head(feature)) * (max_half - minimum_half)
        ranges = torch.stack((center - half, center + half), dim=-1)
        # The training sampler and collision checker operate on mirrored pairs.
        pair = 0.5 * (ranges[:, :3] + ranges[:, 3:])
        return pair[:, (0, 1, 2, 0, 1, 2)]


def generate_dataset(labeler: ApiRangeLabeler, samples: int, seed: int) -> Dict[str, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    low, high = torch.tensor(CONDITION_LOW), torch.tensor(CONDITION_HIGH)
    conditions, ranges, attempts, none_count = [], [], 0, 0
    while len(conditions) < samples:
        attempts += 1
        condition = low + torch.rand(len(CONDITION_NAMES), generator=generator) * (high - low)
        try:
            interval = labeler.label(condition)
        except labeler.no_feasible:
            none_count += 1
            continue
        conditions.append(condition)
        ranges.append(interval)
        if len(conditions) % 100 == 0 or len(conditions) == samples:
            print(f"labels: {len(conditions)}/{samples}, acceptance={len(conditions) / attempts:.1%}")
    return {"conditions": torch.stack(conditions), "ranges": torch.stack(ranges),
            # Explicit aliases make this artifact usable as the runtime
            # valid-envelope bank as well as supervised-training data.
            "valid_conditions": torch.stack(conditions),
            "valid_ranges": torch.stack(ranges),
            "condition_names": CONDITION_NAMES, "leg_names": LEG_NAMES,
            "condition_low": low, "condition_high": high,
            "label_source": LABEL_SOURCE,
            "attempts": attempts, "none_count": none_count}


def train(dataset: Dict[str, Tensor], args) -> Dict[str, object]:
    if dataset.get("label_source") != LABEL_SOURCE:
        raise ValueError(
            "Dataset uses the legacy neutral/trimmed HAA labels. Regenerate it "
            "without --train-only before fitting the directional running-range network."
        )
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
            "label_source": LABEL_SOURCE,
            "running_min_width": MIN_RUNNING_WIDTH,
            # Keep the exact algorithm-approved inputs with the network. RL
            # reset samples only from this bank; it never asks the regressor
            # to invent a running interval for a rejection-range ``none``.
            "valid_conditions": inputs.detach().cpu(),
            "valid_ranges": targets.detach().cpu()}


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
