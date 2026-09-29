"""Capsule-based self-interference check for coupled EL4090 HAA ranges.

The check holds HFE/KFE at ``default_dof_pos`` and evaluates every lower/upper
corner of the three mirrored HAA pairs.  It is conservative with respect to
that finite set of poses and the EL4090 capsule proxy; it is not a replacement
for Isaac Gym's mesh-contact simulation during rollouts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch


_LEGS = ("LB", "LF", "LM", "RB", "RF", "RM")
_PAIR_INDICES = ((0, 3), (1, 4), (2, 5))


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3] / "utils" / "envelop" / "el4090_envelope" / "src"


def _urdf_path() -> Path:
    relative = Path("resources/robots/el_4090/urdf/el_4090.urdf")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find {relative}")


class El4090HaaInterferenceChecker:
    """Reject HAA boxes whose paired-leg corners overlap capsule proxies."""

    def __init__(self, dof_names: Sequence[str], default_dof_pos: torch.Tensor, *, safety_margin: float = 0.02) -> None:
        source_root = _source_root()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from el4090_envelope import BatchedUrdfKinematics, default_el4090_capsules, load_urdf_joints
        from el4090_envelope.model import EL4090_JOINT_NAMES

        dof_index = {name: index for index, name in enumerate(dof_names)}
        if set(EL4090_JOINT_NAMES).difference(dof_index):
            raise ValueError("Isaac HAA DOF names do not match the EL4090 kinematics model")
        self.joint_names = tuple(EL4090_JOINT_NAMES)
        self.haa_indices = torch.tensor(
            [self.joint_names.index(f"{leg}_HAA") for leg in _LEGS], dtype=torch.long,
        )
        self.default_q = torch.tensor(
            [float(default_dof_pos[dof_index[name]]) for name in self.joint_names], dtype=torch.float32,
        )
        self.kinematics = BatchedUrdfKinematics(load_urdf_joints(_urdf_path()))
        self.capsules = tuple(default_el4090_capsules())
        self.links = tuple(capsule.link for capsule in self.capsules)
        self.local_endpoints = torch.tensor([[capsule.start, capsule.end] for capsule in self.capsules])
        self.radii = torch.tensor([capsule.radius for capsule in self.capsules])
        self.safety_margin = float(safety_margin)
        self.pair_indices = self._collision_pair_indices()

    def _collision_pair_indices(self) -> torch.Tensor:
        """Keep torso-vs-distal-leg and inter-leg pairs; skip linked neighbors."""
        pairs = []
        for first, capsule_a in enumerate(self.capsules):
            leg_a = capsule_a.name.split("_", 1)[0] if capsule_a.name != "base_x" else None
            for second, capsule_b in enumerate(self.capsules[:first]):
                leg_b = capsule_b.name.split("_", 1)[0] if capsule_b.name != "base_x" else None
                if leg_a == leg_b:  # same-leg adjacent links are expected to meet.
                    continue
                # Base-to-hip attachment is also an intentional contact.
                if (capsule_a.name == "base_x" and capsule_b.name.endswith("_hip")) or (
                    capsule_b.name == "base_x" and capsule_a.name.endswith("_hip")
                ):
                    continue
                pairs.append((first, second))
        return torch.tensor(pairs, dtype=torch.long)

    @staticmethod
    def _segment_samples(endpoints: torch.Tensor, count: int = 7) -> torch.Tensor:
        weights = torch.linspace(0.0, 1.0, count, dtype=endpoints.dtype, device=endpoints.device)
        return endpoints[..., 0, :].unsqueeze(-2) + weights.view(1, 1, -1, 1) * (
            endpoints[..., 1, :] - endpoints[..., 0, :]
        ).unsqueeze(-2)

    def is_safe(self, ranges: torch.Tensor) -> torch.Tensor:
        """Return one bool per ``[batch, 6, 2]`` coupled HAA interval box."""
        if ranges.ndim != 3 or ranges.shape[1:] != (6, 2):
            raise ValueError(f"Expected ranges [batch, 6, 2], got {tuple(ranges.shape)}")
        if torch.any(ranges[..., 0] >= ranges[..., 1]):
            raise ValueError("All HAA intervals must satisfy lower < upper")
        if not torch.equal(ranges[:, 0], ranges[:, 3]) or not torch.equal(ranges[:, 1], ranges[:, 4]) or not torch.equal(ranges[:, 2], ranges[:, 5]):
            raise ValueError("HAA ranges must be equal for LB/RB, LF/RF, and LM/RM")

        device, dtype = ranges.device, ranges.dtype
        masks = torch.tensor([[bit >> pair & 1 for pair in range(3)] for bit in range(8)], device=device)
        pair_ranges = ranges[:, (0, 1, 2), :]  # [batch, 3, lower/upper]
        values = torch.gather(pair_ranges.unsqueeze(1).expand(-1, 8, -1, -1), 3, masks.view(1, 8, 3, 1).expand(ranges.shape[0], -1, -1, -1)).squeeze(-1)
        q = self.default_q.to(device=device, dtype=dtype).view(1, 1, -1).expand(ranges.shape[0], 8, -1).clone()
        q[..., self.haa_indices.to(device)] = values[:, :, (0, 1, 2, 0, 1, 2)]
        q = q.reshape(-1, q.shape[-1])

        endpoints = self.kinematics.points(q, self.links, self.local_endpoints.to(device=device, dtype=dtype))
        points = self._segment_samples(endpoints)
        pair_indices = self.pair_indices.to(device)
        first = points[:, pair_indices[:, 0]]
        second = points[:, pair_indices[:, 1]]
        squared_distance = torch.square(first.unsqueeze(-2) - second.unsqueeze(-3)).sum(dim=-1)
        distance = torch.sqrt(squared_distance.amin(dim=(-1, -2)).clamp_min(1e-12))
        required = self.radii.to(device=device, dtype=dtype)[pair_indices].sum(dim=1) + self.safety_margin
        collision = (distance <= required.unsqueeze(0)).any(dim=1)
        return (~collision).reshape(ranges.shape[0], 8).all(dim=1)


def main() -> None:
    """Sample and validate direct, paired HAA boxes without Isaac Gym."""
    # Keep this standalone: importing ``legged_gym.envs`` would require Isaac
    # Gym even though the kinematics/capsule validation itself does not.
    from haa_range_state import HaaRangeState

    class SamplingCfg:
        leg_names = list(_LEGS)
        joint_limits = [-1.3, 1.3]
        min_width = 0.3
        max_sampling_attempts = 16

    dof_names = [
        f"{leg}_{joint}"
        for leg in _LEGS
        for joint in ("HAA", "HFE", "KFE")
    ]
    checker = El4090HaaInterferenceChecker(dof_names, torch.zeros(len(dof_names)))
    batch_size, rounds = 256, 10
    sampler = HaaRangeState(SamplingCfg, batch_size, "cpu", interference_checker=checker)

    total = 0
    observed_width = float("inf")
    for round_index in range(rounds):
        ranges = sampler.sample(torch.arange(batch_size))
        assert torch.all((-1.3 <= ranges) & (ranges <= 1.3))
        assert torch.all(ranges[..., 1] - ranges[..., 0] >= 0.3 - 1e-6)
        assert torch.equal(ranges[:, 0], ranges[:, 3])  # LB/RB
        assert torch.equal(ranges[:, 1], ranges[:, 4])  # LF/RF
        assert torch.equal(ranges[:, 2], ranges[:, 5])  # LM/RM
        assert torch.all(checker.is_safe(ranges))
        observed_width = min(observed_width, float((ranges[..., 1] - ranges[..., 0]).min()))
        total += batch_size
        print(f"round {round_index + 1:2d}/{rounds}: {batch_size} valid coupled HAA boxes")

    # Deliberately adversarial but valid sampled box: all three paired leg
    # groups swing close to the -1.3 rad end stop. It meets the endpoint,
    # width, and left/right constraints, but folds multiple legs through the
    # same side of the body and must be rejected by the interference check.
    likely_interference = torch.tensor([[
        [-1.3, -1.0],  # LB
        [-1.3, -1.0],  # LF
        [-1.3, -1.0],  # LM
        [-1.3, -1.0],  # RB == LB
        [-1.3, -1.0],  # RF == LF
        [-1.3, -1.0],  # RM == LM
    ]])
    assert not checker.is_safe(likely_interference).item()

    full_range = torch.tensor([[[-1.3, 1.3]] * 6])
    assert not checker.is_safe(full_range).item()
    print(
        f"passed: {total} sampled boxes; minimum observed width={observed_width:.4f} rad; "
        "the likely-interference and full-range boxes were correctly rejected"
    )


if __name__ == "__main__":
    main()
