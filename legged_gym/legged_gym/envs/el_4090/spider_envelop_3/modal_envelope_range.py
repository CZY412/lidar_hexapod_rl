"""HAA running ranges for a fixed spider-default EL4090 posture."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch


_GEOMETRY_NAMES = ("front_width", "middle_width", "back_width", "forward_limit", "backward_limit")
_FORWARD_SPLAY_LEGS = {"LF", "RF"}


class NoFeasibleHaaEnvelopeError(ValueError):
    """The allowed polygon has no valid HAA posture with spider HFE/KFE pins."""


def _package_source_root() -> Path:
    return Path(__file__).resolve().parents[3] / "utils" / "envelop" / "el4090_envelope" / "src"


def _find_urdf() -> Path:
    relative = Path("resources/robots/el_4090/urdf/el_4090.urdf")
    for root in Path(__file__).resolve().parents:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find {relative} above {__file__}")


class SpiderEnvelopeHaaRangeEstimator:
    """Compute API-safe HAA intervals while HFE/KFE stay at spider defaults.

    All 18 pinned joints start at ``spider_dof_pos``. The six HAA joints are
    swept only inside supplied spider-to-mammal search ranges; HFE and KFE are
    never interpolated or sampled.
    """

    def __init__(self, cfg, *, condition_names: Sequence[str], leg_names: Sequence[str],
                 dof_names: Sequence[str], spider_dof_pos: torch.Tensor,
                 haa_search_ranges: Sequence[Sequence[float]]) -> None:
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
        self._haa_indices = tuple(self._joint_names.index(f"{leg}_HAA") for leg in self._leg_names)
        if len(haa_search_ranges) != len(self._leg_names):
            raise ValueError("haa_search_ranges must provide one [lower, upper] pair per HAA leg")
        self._haa_search_ranges = tuple((float(lower), float(upper)) for lower, upper in haa_search_ranges)
        if any(lower > upper for lower, upper in self._haa_search_ranges):
            raise ValueError("Each HAA search range must have lower <= upper")
        self._kinematics = BatchedUrdfKinematics(load_urdf_joints(_find_urdf()))
        self._capsules = tuple(default_el4090_capsules())
        self._directions = support_directions(int(cfg.directions))
        self._haa_rejection_ranges = haa_rejection_ranges
        self._capsule_support = capsule_support
        self._margin = float(cfg.margin)
        self._tolerance = float(cfg.tolerance)
        self._sweep_steps = int(cfg.sweep_steps)
        self._cache_precision = int(cfg.cache_precision)
        self._cache: Dict[Tuple[float, ...], torch.Tensor] = {}

    @property
    def directions(self) -> torch.Tensor:
        return self._directions

    def occupied_support_from_isaac_dof(self, dof_pos: torch.Tensor, dof_names: Sequence[str]) -> torch.Tensor:
        dof_index = {name: index for index, name in enumerate(dof_names)}
        q = torch.tensor([float(dof_pos[dof_index[name]]) for name in self._joint_names], dtype=torch.float32)
        return self._capsule_support(self._kinematics, q.unsqueeze(0), self._capsules, self._directions)[0]

    def allowed_support(self, condition: torch.Tensor) -> torch.Tensor:
        front_width, middle_width, back_width, forward_limit, backward_limit = (
            condition[self._condition_index[name]] for name in _GEOMETRY_NAMES
        )
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

    @staticmethod
    def _select_safe_interval(leg_name: str, candidates):
        """Choose spider zero, otherwise the leg group's preferred splay side."""
        containing_zero = [item for item in candidates if item[0] <= 0.0 <= item[1]]
        if containing_zero:
            return max(containing_zero, key=lambda item: item[1] - item[0])
        centers = [0.5 * (item[0] + item[1]) for item in candidates]
        if leg_name in _FORWARD_SPLAY_LEGS:
            # Front legs splay forward at negative HAA in the EL4090 convention.
            return candidates[centers.index(min(centers))]
        # Middle/rear legs splay rearward at positive HAA.
        return candidates[centers.index(max(centers))]

    def _compute_one(self, condition: torch.Tensor) -> torch.Tensor:
        lower, upper = self._kinematics.joint_limits(dtype=torch.float32)
        for index, (search_lower, search_upper) in zip(self._haa_indices, self._haa_search_ranges):
            lower[index] = search_lower
            upper[index] = search_upper
        api_result = self._haa_rejection_ranges(
            self._kinematics, self._capsules, self._directions, self.allowed_support(condition),
            lower, upper, self._spider_q, self._haa_indices,
            tolerance=self._tolerance, sweep_steps=self._sweep_steps,
        )
        ranges = []
        for leg_name, index in zip(self._leg_names, self._haa_indices):
            candidates = self._complement(api_result.per_haa_joint_intervals[index], float(lower[index]), float(upper[index]))
            if not candidates:
                values = ", ".join(f"{value:.3f}" for value in condition.tolist())
                raise NoFeasibleHaaEnvelopeError(
                    "Allowed envelope has no feasible HAA posture with fixed spider HFE/KFE "
                    f"(leg={leg_name}, api_mode={api_result.mode}, geometry=[{values}])."
                )
            ranges.append(self._select_safe_interval(leg_name, candidates))
        return torch.tensor(ranges, dtype=torch.float32)

    def compute(self, conditions: torch.Tensor) -> torch.Tensor:
        if conditions.ndim != 2:
            raise ValueError(f"Expected [batch, condition_dim], got {tuple(conditions.shape)}")
        result = []
        for condition in conditions.detach().to(device="cpu", dtype=torch.float32):
            key = tuple(round(float(value), self._cache_precision) for value in condition.tolist())
            cached = self._cache.get(key)
            if cached is None:
                cached = self._compute_one(condition)
                self._cache[key] = cached
            result.append(cached)
        return torch.stack(result).to(device=conditions.device, dtype=conditions.dtype)

    def __call__(self, conditions: torch.Tensor) -> torch.Tensor:
        return self.compute(conditions)
