"""Bridge the five-parameter allowed envelope to modal HAA ranges."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch


_LEG_TO_PRIOR_COLUMN = {"LF": 0, "RF": 0, "LM": 1, "RM": 1, "LB": 2, "RB": 2}
_GEOMETRY_NAMES = ("front_width", "middle_width", "back_width", "forward_limit", "backward_limit")


def _package_source_root() -> Path:
    return Path(__file__).resolve().parents[3] / "utils" / "envelop" / "el4090_envelope" / "src"


def _find_urdf() -> Path:
    relative = Path("resources/robots/el_4090/urdf/el_4090.urdf")
    for root in Path(__file__).resolve().parents:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find {relative} above {__file__}")


class ModalEnvelopeHaaRangeEstimator:
    """Use the new API to derive HAA ranges and morphology priors.

    The external condition is the C-mode hexagonal allowed envelope only.
    Safe HAA interval centers are converted to front/middle/back interpolation
    factors, which then determine all HAA/HFE/KFE target joints for the next
    API query.
    """

    def __init__(self, cfg, *, condition_names: Sequence[str], leg_names: Sequence[str],
                 dof_names: Sequence[str], spider_dof_pos: torch.Tensor,
                 mammal_dof_pos: torch.Tensor) -> None:
        source_root = _package_source_root()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from el4090_envelope import (EL4090_JOINT_NAMES, BatchedUrdfKinematics,
                                     default_el4090_capsules, haa_rejection_ranges,
                                     load_urdf_joints, support_directions)
        from el4090_envelope.model import capsule_support

        self._joint_names = tuple(EL4090_JOINT_NAMES)
        self._leg_names = tuple(leg_names)
        self._condition_index = {name: index for index, name in enumerate(condition_names)}
        missing = set(_GEOMETRY_NAMES).difference(self._condition_index)
        if missing:
            raise ValueError(f"Missing allowed-envelope entries: {sorted(missing)}")
        dof_index = {name: index for index, name in enumerate(dof_names)}
        if set(self._joint_names).difference(dof_index):
            raise ValueError("Isaac DOF names do not match the EL4090 envelope joint names")
        self._spider_q = torch.tensor([float(spider_dof_pos[dof_index[name]]) for name in self._joint_names], dtype=torch.float32)
        self._mammal_q = torch.tensor([float(mammal_dof_pos[dof_index[name]]) for name in self._joint_names], dtype=torch.float32)
        self._haa_indices = tuple(self._joint_names.index(f"{leg}_HAA") for leg in self._leg_names)
        self._kinematics = BatchedUrdfKinematics(load_urdf_joints(_find_urdf()))
        self._capsules = tuple(default_el4090_capsules())
        self._directions = support_directions(int(cfg.directions))
        self._haa_rejection_ranges = haa_rejection_ranges
        self._capsule_support = capsule_support
        self._margin = float(cfg.margin)
        self._tolerance = float(cfg.tolerance)
        self._sweep_steps = int(cfg.sweep_steps)
        self._prior_iterations = int(cfg.prior_iterations)
        self._cache_precision = int(cfg.cache_precision)
        self._cache: Dict[Tuple[float, ...], Tuple[torch.Tensor, torch.Tensor]] = {}

    def occupied_support_from_isaac_dof(self, dof_pos: torch.Tensor, dof_names: Sequence[str]) -> torch.Tensor:
        """Return the capsule occupied support for one simulator posture."""
        dof_index = {name: index for index, name in enumerate(dof_names)}
        q = torch.tensor([float(dof_pos[dof_index[name]]) for name in self._joint_names], dtype=torch.float32)
        return self._capsule_support(self._kinematics, q.unsqueeze(0), self._capsules, self._directions)[0]

    @property
    def directions(self) -> torch.Tensor:
        return self._directions

    def _value(self, condition: torch.Tensor, name: str) -> torch.Tensor:
        return condition[self._condition_index[name]]

    def allowed_support(self, condition: torch.Tensor) -> torch.Tensor:
        """C-mode hexagon support, expanded by the configured safety margin."""
        front_width = self._value(condition, "front_width")
        middle_width = self._value(condition, "middle_width")
        back_width = self._value(condition, "back_width")
        forward_limit = self._value(condition, "forward_limit")
        backward_limit = self._value(condition, "backward_limit")
        vertices = torch.stack((
            torch.stack((forward_limit, front_width)),
            torch.stack((torch.zeros_like(middle_width), middle_width)),
            torch.stack((backward_limit, back_width)),
            torch.stack((backward_limit, -back_width)),
            torch.stack((torch.zeros_like(middle_width), -middle_width)),
            torch.stack((forward_limit, -front_width)),
        ))
        return (self._directions @ vertices.T).amax(dim=1) + self._margin

    @staticmethod
    def _complement(rejected, lower: float, upper: float):
        cursor, allowed = lower, []
        for start, end in sorted(rejected):
            start, end = max(lower, float(start)), min(upper, float(end))
            if start > cursor:
                allowed.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < upper:
            allowed.append((cursor, upper))
        return allowed

    def _target_q(self, priors: torch.Tensor) -> torch.Tensor:
        q = self._spider_q.clone()
        for leg_name in self._leg_names:
            prior = priors[_LEG_TO_PRIOR_COLUMN[leg_name]]
            for kind in ("HAA", "HFE", "KFE"):
                index = self._joint_names.index(f"{leg_name}_{kind}")
                q[index] = self._spider_q[index] + prior * (self._mammal_q[index] - self._spider_q[index])
        return q

    def _ranges_for(self, condition: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        q = self._target_q(priors)
        lower, upper = self._kinematics.joint_limits(dtype=torch.float32)
        for index in self._haa_indices:
            lower[index] = torch.minimum(self._spider_q[index], self._mammal_q[index])
            upper[index] = torch.maximum(self._spider_q[index], self._mammal_q[index])
        api_result = self._haa_rejection_ranges(
            self._kinematics, self._capsules, self._directions, self.allowed_support(condition),
            lower, upper, q, self._haa_indices, tolerance=self._tolerance, sweep_steps=self._sweep_steps,
        )
        ranges = []
        for leg_name, index in zip(self._leg_names, self._haa_indices):
            candidates = self._complement(api_result.per_haa_joint_intervals[index], float(lower[index]), float(upper[index]))
            if not candidates:
                raise RuntimeError(f"No feasible HAA interval for {leg_name}; API mode={api_result.mode}")
            target = float(q[index])
            containing = [candidate for candidate in candidates if candidate[0] <= target <= candidate[1]]
            ranges.append(max(containing or candidates, key=lambda candidate: candidate[1] - candidate[0]))
        return torch.tensor(ranges, dtype=torch.float32)

    def _priors_from_ranges(self, ranges: torch.Tensor) -> torch.Tensor:
        groups = [[] for _ in range(3)]
        for row, (leg_name, index) in enumerate(zip(self._leg_names, self._haa_indices)):
            denominator = self._mammal_q[index] - self._spider_q[index]
            if abs(float(denominator)) > 1e-6:
                center = 0.5 * (ranges[row, 0] + ranges[row, 1])
                groups[_LEG_TO_PRIOR_COLUMN[leg_name]].append((center - self._spider_q[index]) / denominator)
        return torch.stack([torch.clamp(torch.stack(group).mean(), 0.0, 1.0) if group else torch.tensor(0.0) for group in groups])

    def _compute_one(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        priors = torch.zeros(3, dtype=torch.float32)
        ranges = self._ranges_for(condition, priors)
        for _ in range(max(1, self._prior_iterations)):
            updated_priors = self._priors_from_ranges(ranges)
            if torch.allclose(updated_priors, priors, atol=1e-4):
                priors = updated_priors
                break
            priors = updated_priors
            ranges = self._ranges_for(condition, priors)
        return ranges, priors

    def compute(self, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if conditions.ndim != 2:
            raise ValueError(f"Expected [batch, condition_dim], got {tuple(conditions.shape)}")
        ranges, priors = [], []
        for condition in conditions.detach().to(device="cpu", dtype=torch.float32):
            key = tuple(round(float(value), self._cache_precision) for value in condition.tolist())
            cached = self._cache.get(key)
            if cached is None:
                cached = self._compute_one(condition)
                self._cache[key] = cached
            range_value, prior_value = cached
            ranges.append(range_value)
            priors.append(prior_value)
        return (torch.stack(ranges).to(device=conditions.device, dtype=conditions.dtype),
                torch.stack(priors).to(device=conditions.device, dtype=conditions.dtype))

    def __call__(self, conditions: torch.Tensor) -> torch.Tensor:
        """Compatibility shorthand returning only HAA ranges."""
        return self.compute(conditions)[0]
