"""External five-parameter allowed-envelope state for envelop_3."""

from __future__ import annotations

from typing import Optional, Sequence

import torch


class AllowedEnvelopeState:
    """Sample or set the five external allowed-envelope geometry values."""

    def __init__(self, cfg, num_envs: int, device) -> None:
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.condition_names = tuple(cfg.condition_names)
        self.condition_dim = len(self.condition_names)
        self.low = torch.tensor(
            [getattr(cfg.ranges, name)[0] for name in self.condition_names],
            dtype=torch.float, device=self.device,
        )
        self.high = torch.tensor(
            [getattr(cfg.ranges, name)[1] for name in self.condition_names],
            dtype=torch.float, device=self.device,
        )
        self.condition = (0.5 * (self.low + self.high)).repeat(self.num_envs, 1)

    def get(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.condition if env_ids is None else self.condition[env_ids]

    def sample(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() != 0:
            random_values = torch.rand((env_ids.numel(), self.condition_dim), device=self.device)
            self.condition[env_ids] = self.low + random_values * (self.high - self.low)
        return self.condition[env_ids]

    def set(
        self,
        values: torch.Tensor | Sequence[Sequence[float]],
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None else env_ids.to(device=self.device, dtype=torch.long)
        )
        values = torch.as_tensor(values, dtype=torch.float, device=self.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape != (target_ids.numel(), self.condition_dim):
            raise ValueError(f"Expected {(target_ids.numel(), self.condition_dim)}, got {tuple(values.shape)}")
        values = torch.maximum(self.low, torch.minimum(values, self.high))
        self.condition[target_ids] = values
        return values
