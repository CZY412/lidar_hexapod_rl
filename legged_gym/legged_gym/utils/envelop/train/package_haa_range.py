"""Geometry teacher for a drop-in EL4090 HAA-range checkpoint.

The deployed ``HaaRangeNetwork`` has a deliberately small, stable contract:
an eight-dimensional legacy envelope/morphology condition in and six ordered
``[lower, upper]`` HAA intervals out.  This module preserves that contract,
but makes labels with :mod:`el4090_envelope`'s capsule-support feasibility
model instead of the old lateral-width approximation.

The package-level geometry is evaluated from the full finite-normal support
vector.  The compact condition is *only* the student input; it must not be
used as the geometry oracle for an asymmetric obstacle envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import torch
from torch import Tensor

try:  # Package import when called from the simulator tree.
    from ..network.haa_swing_range import HaaRangeConfig
except ImportError:  # Direct-file execution: do not import legged_gym.utils.
    from _standalone import load_haa_range_module
    HaaRangeConfig = load_haa_range_module().HaaRangeConfig


_PACKAGE_SRC = Path(__file__).resolve().parents[5] / "el4090_envelope" / "src"
if str(_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SRC))

from el4090_envelope import (  # noqa: E402
    EL4090_JOINT_NAMES,
    BatchedUrdfKinematics,
    add_support_margin,
    default_el4090_capsules,
    haa_rejection_ranges,
    load_urdf_joints,
    support_directions,
)


LEGACY_HAA_ORDER: tuple[str, ...] = ("RF", "RM", "RB", "LF", "LM", "LB")
PACKAGE_HAA_ORDER: tuple[str, ...] = ("LB", "LF", "LM", "RB", "RF", "RM")


@dataclass(frozen=True)
class PackageHaaTeacherConfig:
    """Numerical choices used to create reproducible geometric labels."""

    directions: int = 32
    tolerance: float = 1e-5
    sweep_steps: int = 161
    fold_samples: int = 2048
    fold_bins: int = 129
    min_rejected_span: float = 0.08
    # Existing 5-D conditions describe a foot footprint, not the occupied
    # capsule outline.  This is an outward offset used only by the optional
    # synthetic condition sampler.  Real planners should provide the true
    # allowed_support directly and leave this path unused.
    synthetic_body_margin: float = 0.45
    output_margin: float = 0.03
    # A rejection complement may contain disconnected HAA folds.  The deployed
    # [center, half] representation may select only the morphology branch.
    require_morphology_branch: bool = True


def build_morphology_reference(condition: Tensor, range_config: HaaRangeConfig) -> Tensor:
    """Build the same preferred HAA centre used by the legacy morphology model.

    ``condition`` must already contain derived morphology priors.  HFE/KFE are
    held at the spider nominal pose: their values are pins for HAA rejection,
    not outputs of the HAA model.
    """
    if condition.ndim != 1 or condition.numel() != len(range_config.condition_names):
        raise ValueError("condition must be one eight-dimensional row")
    result = torch.zeros(18, dtype=condition.dtype, device=condition.device)
    priors = {
        "F": condition[range_config.condition_names.index("morphology_front_prior")],
        "M": condition[range_config.condition_names.index("morphology_middle_prior")],
        "B": condition[range_config.condition_names.index("morphology_back_prior")],
    }
    spider = dict(zip(range_config.leg_names, range_config.spider_haa))
    mammal = dict(zip(range_config.leg_names, range_config.mammal_haa))
    for leg in PACKAGE_HAA_ORDER:
        joint = EL4090_JOINT_NAMES.index(f"{leg}_HAA")
        prior = priors[leg[1]]
        result[joint] = float(spider[leg]) + prior * (float(mammal[leg]) - float(spider[leg]))
        # Stable nominal sagittal pose; this deliberately matches the deployed
        # Envelop2 spider pose rather than sampling unrelated HFE/KFE values.
        result[EL4090_JOINT_NAMES.index(f"{leg}_HFE")] = 0.4036
        result[EL4090_JOINT_NAMES.index(f"{leg}_KFE")] = -0.55019
    return result


def condition_footprint_support(
    condition: Tensor,
    directions: Tensor,
    condition_names: Sequence[str],
    *,
    body_margin: float,
) -> Tensor:
    """Turn a symmetric legacy footprint into finite-normal allowed support.

    This compatibility constructor is appropriate for generating a first
    checkpoint from the current five geometric condition fields.  For live
    obstacle envelopes, pass planner-generated ``allowed_support`` to the
    teacher instead; reducing it to this footprint would discard asymmetry.
    """
    if condition.shape[-1] != len(condition_names):
        raise ValueError("condition width does not match condition_names")
    index = {name: i for i, name in enumerate(condition_names)}
    front = condition[..., index["forward_limit"]]
    back = condition[..., index["backward_limit"]]
    widths = torch.stack((
        condition[..., index["front_width"]],
        condition[..., index["middle_width"]],
        condition[..., index["back_width"]],
    ), dim=-1)
    # Convex hull of front/middle/back lateral anchors.  It exactly captures
    # the symmetric piecewise-linear footprint represented by the old fields.
    vertices = torch.stack((
        torch.stack((front, widths[..., 0]), dim=-1),
        torch.stack((front, -widths[..., 0]), dim=-1),
        torch.stack((torch.zeros_like(front), widths[..., 1]), dim=-1),
        torch.stack((torch.zeros_like(front), -widths[..., 1]), dim=-1),
        torch.stack((back, widths[..., 2]), dim=-1),
        torch.stack((back, -widths[..., 2]), dim=-1),
    ), dim=-2)
    support = torch.einsum("...vd,kd->...vk", vertices, directions).amax(dim=-2)
    return add_support_margin(support, body_margin)


def _accessible_intervals(
    rejected: Iterable[tuple[float, float]], lower: float, upper: float,
) -> list[tuple[float, float]]:
    cursor = lower
    accessible: list[tuple[float, float]] = []
    for start, end in sorted(rejected):
        start, end = max(lower, start), min(upper, end)
        if start > cursor:
            accessible.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < upper:
        accessible.append((cursor, upper))
    return [(start, end) for start, end in accessible if end > start]


def _choose_prior_interval(
    intervals: Sequence[tuple[float, float]],
    prior: float,
    *,
    require_containing: bool,
) -> tuple[float, float]:
    """Choose one connected feasible band without bridging rejected bands.

    Requiring containment is the safe runtime policy: it never substitutes an
    alternate fold when the morphology branch has been blocked.
    """
    if not intervals:
        raise RuntimeError("No accessible HAA interval for this envelope")
    containing = [interval for interval in intervals if interval[0] <= prior <= interval[1]]
    if require_containing:
        if not containing:
            raise RuntimeError("Morphology HAA branch is infeasible; alternate fold is rejected")
        return max(containing, key=lambda interval: interval[1] - interval[0])
    def key(interval: tuple[float, float]) -> tuple[float, float]:
        low, high = interval
        distance = 0.0 if low <= prior <= high else min(abs(prior - low), abs(prior - high))
        return distance, -(high - low)
    return min(intervals, key=key)


class PackageHaaRangeTeacher:
    """CPU/offline capsule-envelope teacher returning legacy-order HAA ranges."""

    def __init__(
        self,
        range_config: HaaRangeConfig,
        *,
        config: PackageHaaTeacherConfig = PackageHaaTeacherConfig(),
        urdf_path: Path | str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.range_config = range_config
        self.config = config
        self.device = torch.device(device)
        if urdf_path is None:
            urdf_path = Path(__file__).resolve().parents[4] / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"
        self.kinematics = BatchedUrdfKinematics(load_urdf_joints(urdf_path))
        self.capsules = default_el4090_capsules()
        self.directions = support_directions(config.directions, dtype=torch.float32, device=self.device)
        lower, upper = self.kinematics.joint_limits(dtype=torch.float32, device=self.device)
        self.lower, self.upper = lower, upper
        self.haa_indices = [EL4090_JOINT_NAMES.index(f"{leg}_HAA") for leg in PACKAGE_HAA_ORDER]

    @torch.no_grad()
    def label(self, condition: Tensor, allowed_support: Tensor) -> Tensor:
        """Return conservative connected ``[6,2]`` ranges in legacy order.

        An invalid envelope has no safe range and raises instead of silently
        returning mechanical limits.  Dataset generation records and drops
        these rows, which prevents unsafe labels from reaching the student.
        """
        condition = condition.to(device=self.device, dtype=torch.float32)
        allowed_support = allowed_support.to(device=self.device, dtype=torch.float32)
        if allowed_support.shape != (self.config.directions,):
            raise ValueError(f"allowed_support must have shape ({self.config.directions},)")
        pinned = build_morphology_reference(condition, self.range_config)
        rejection = haa_rejection_ranges(
            self.kinematics, self.capsules, self.directions, allowed_support,
            self.lower, self.upper, pinned, self.haa_indices,
            tolerance=self.config.tolerance,
            sweep_steps=self.config.sweep_steps,
            fold_samples=self.config.fold_samples,
            fold_bins=self.config.fold_bins,
            min_rej_span=self.config.min_rejected_span,
        )
        if rejection.mode == "none":
            raise RuntimeError("Envelope admits no sampled HAA fold at the morphology pins")
        package_ranges: dict[str, tuple[float, float]] = {}
        for leg, joint_index in zip(PACKAGE_HAA_ORDER, self.haa_indices):
            intervals = _accessible_intervals(
                rejection.per_haa_joint_intervals[joint_index],
                float(self.lower[joint_index]), float(self.upper[joint_index]),
            )
            preferred = float(pinned[joint_index])
            low, high = _choose_prior_interval(
                intervals, preferred,
                require_containing=self.config.require_morphology_branch,
            )
            # Keep the legacy morphology swing allowance as a hard local
            # branch window. Geometry determines endpoints inside it, but a
            # large free envelope cannot bridge to the opposite folded pose.
            morphology_region = {"F": "front", "M": "middle", "B": "back"}[leg[1]]
            prior_name = f"morphology_{morphology_region}_prior"
            prior = float(condition[self.range_config.condition_names.index(prior_name)])
            morphology_half = (
                self.range_config.spider_swing_limit
                + prior * (self.range_config.mammal_swing_limit - self.range_config.spider_swing_limit)
            )
            low = max(low, preferred - morphology_half)
            high = min(high, preferred + morphology_half)
            # Do not train the student to reproduce a boundary contact exactly.
            low += self.config.output_margin
            high -= self.config.output_margin
            if high <= low:
                raise RuntimeError(f"Feasible band for {leg} vanished after output_margin")
            package_ranges[leg] = (low, high)
        return torch.tensor(
            [package_ranges[leg] for leg in LEGACY_HAA_ORDER],
            dtype=torch.float32,
            device=self.device,
        )
