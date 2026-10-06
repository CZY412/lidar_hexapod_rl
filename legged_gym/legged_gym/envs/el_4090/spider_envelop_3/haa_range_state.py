"""Coupled, directly sampled and interference-checked HAA intervals."""

from __future__ import annotations

from typing import Optional, Sequence

import torch


class HaaRangeState:
    """Own valid directly sampled ``[lower, upper]`` HAA intervals."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        device,
        *,
        interference_checker=None,
        interference_check_batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.leg_names = tuple(cfg.leg_names)
        self.joint_low = float(cfg.joint_limits[0])
        self.joint_high = float(cfg.joint_limits[1])
        self.min_width = float(cfg.min_width)
        self.max_attempts = int(cfg.max_sampling_attempts)
        self.interference_checker = interference_checker
        self.interference_check_batch_size = int(interference_check_batch_size)
        size = len(self.leg_names)
        if size != 6 or self.leg_names != ("LB", "LF", "LM", "RB", "RF", "RM"):
            raise ValueError("haa_range leg_names must be LB, LF, LM, RB, RF, RM")
        if self.joint_low >= self.joint_high or self.min_width < 0.2 or self.min_width > self.joint_high - self.joint_low:
            raise ValueError("haa_range must be within valid joint bounds and have min_width >= 0.2")
        if self.max_attempts < 1:
            raise ValueError("haa_range.max_sampling_attempts must be positive")
        if self.interference_check_batch_size < 1:
            raise ValueError("interference_check_batch_size must be positive")
        self.ranges = torch.empty(self.num_envs, size, 2, dtype=torch.float, device=self.device)
        self.ranges[..., 0] = -0.5 * self.min_width
        self.ranges[..., 1] = 0.5 * self.min_width

    def get(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.ranges if env_ids is None else self.ranges[env_ids]

    def sample(self, env_ids: torch.Tensor) -> torch.Tensor:
        # Training uses direct sampling. The optional checker is retained for
        # offline validation, where its all-corner capsule test is affordable.
        if self.interference_checker is None:
            self.ranges[env_ids] = self._sample_candidate(env_ids.numel())
            return self.ranges[env_ids]
        remaining = env_ids
        for _ in range(self.max_attempts):
            if not remaining.numel():
                break
            candidate = self._sample_candidate(remaining.numel())
            safe = self._is_safe(candidate)
            if safe.any():
                self.ranges[remaining[safe]] = candidate[safe]
            remaining = remaining[~safe]
        if remaining.numel():
            fallback = self._fallback(remaining.numel())
            if not torch.all(self._is_safe(fallback)):
                raise RuntimeError("The configured fallback HAA interval fails the self-interference check")
            self.ranges[remaining] = fallback
        return self.ranges[env_ids]

    def _is_safe(self, ranges: torch.Tensor) -> torch.Tensor:
        """Run the expensive all-corner check in bounded-size GPU batches."""
        if self.interference_checker is None:
            return torch.ones(ranges.shape[0], dtype=torch.bool, device=self.device)
        batch_size = self.interference_check_batch_size
        if ranges.shape[0] <= batch_size:
            return self.interference_checker.is_safe(ranges)
        return torch.cat([
            self.interference_checker.is_safe(ranges[start:start + batch_size])
            for start in range(0, ranges.shape[0], batch_size)
        ], dim=0)

    def _sample_candidate(self, count: int) -> torch.Tensor:
        # Sample the three left/right leg-pair intervals. Each lower endpoint
        # is chosen first, then its upper endpoint; this enforces min_width.
        lower = self.joint_low + torch.rand(count, 3, device=self.device) * (
            self.joint_high - self.joint_low - self.min_width
        )
        upper = lower + self.min_width + torch.rand(count, 3, device=self.device) * (
            self.joint_high - lower - self.min_width
        )
        pairs = torch.stack((lower, upper), dim=-1)
        return pairs[:, (0, 1, 2, 0, 1, 2)]

    def _fallback(self, count: int) -> torch.Tensor:
        # Symmetric neutral box around the unique default HAA posture (zero).
        half = 0.5 * self.min_width
        return torch.tensor([[-half, half]] * 6, dtype=torch.float, device=self.device).expand(count, -1, -1).clone()

    def set(self, values: torch.Tensor | Sequence[Sequence[Sequence[float]]], env_ids=None) -> torch.Tensor:
        target_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids.to(self.device)
        values = torch.as_tensor(values, dtype=torch.float, device=self.device)
        expected = (target_ids.numel(), len(self.leg_names), 2)
        if values.shape != expected:
            raise ValueError(f"Expected HAA ranges {expected}, got {tuple(values.shape)}")
        if torch.any(values < self.joint_low) or torch.any(values > self.joint_high):
            raise ValueError(f"HAA range endpoints must stay within [{self.joint_low}, {self.joint_high}]")
        if torch.any(values[..., 1] - values[..., 0] < self.min_width):
            raise ValueError(f"Each HAA range must have width >= {self.min_width}")
        if not (torch.equal(values[:, 0], values[:, 3]) and torch.equal(values[:, 1], values[:, 4]) and torch.equal(values[:, 2], values[:, 5])):
            raise ValueError("LB/RB, LF/RF, and LM/RM must use identical HAA ranges")
        if not torch.all(self._is_safe(values)):
            raise ValueError("Requested HAA ranges fail the self-interference check")
        self.ranges[target_ids] = values
        return self.ranges[target_ids]
