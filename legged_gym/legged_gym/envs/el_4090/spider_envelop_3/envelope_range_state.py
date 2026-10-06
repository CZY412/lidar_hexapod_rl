"""Five-parameter envelope sampling backed by the distilled HAA-range net."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .haa_range_state import HaaRangeState


class ApiHaaRangeNetwork(nn.Module):
    """Checkpoint-compatible five-input, six-interval regressor."""

    def __init__(self, hidden_dims=(256, 256, 128)) -> None:
        super().__init__()
        layers, width = [], 5
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.ELU()]
            width = hidden
        self.backbone = nn.Sequential(*layers)
        self.center_head, self.half_head = nn.Linear(width, 6), nn.Linear(width, 6)
        self.register_buffer("condition_low", torch.tensor((.3, .3, .3, .6, -.9)))
        self.register_buffer("condition_high", torch.tensor((.6, .7, .6, .9, -.6)))
        self.register_buffer("haa_low", torch.tensor((0., -1.5, 0., 0., -1.5, 0.)))
        self.register_buffer("haa_high", torch.tensor((1.5, 0., 1.5, 1.5, 0., 1.5)))

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        normalized = 2.0 * (condition - self.condition_low) / (self.condition_high - self.condition_low) - 1.0
        feature, min_half = self.backbone(normalized), .15
        center = self.haa_low + min_half + torch.sigmoid(self.center_head(feature)) * (self.haa_high - self.haa_low - 2 * min_half)
        maximum = torch.minimum(center - self.haa_low, self.haa_high - center)
        half = min_half + torch.sigmoid(self.half_head(feature)) * (maximum - min_half)
        raw = torch.stack((center - half, center + half), dim=-1)
        paired = .5 * (raw[:, :3] + raw[:, 3:])
        return paired[:, (0, 1, 2, 0, 1, 2)]


class EnvelopeRangeState(HaaRangeState):
    """Freely samples five envelope parameters, then predicts HAA ranges.

    Collision filtering is deliberately offline-only.  Runtime retries solely
    when the fitted network returns an invalid/narrow range.
    """

    condition_names = ("front_width", "middle_width", "back_width", "forward_limit", "backward_limit")

    def __init__(self, cfg, num_envs, device, *, checkpoint: str):
        super().__init__(cfg, num_envs, device, interference_checker=None,
                         interference_check_batch_size=1)
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"HAA range checkpoint not found: {path}. Train it with train_api_haa_range.py first.")
        payload = torch.load(path, map_location="cpu")
        if tuple(payload.get("condition_names", ())) != self.condition_names or tuple(payload.get("output_names", ())) != ("lower", "upper"):
            raise ValueError(f"{path} is not a five-parameter HAA-range checkpoint")
        expected_low = torch.tensor((0., -1.5, 0., 0., -1.5, 0.))
        expected_high = torch.tensor((1.5, 0., 1.5, 1.5, 0., 1.5))
        if not (
            torch.equal(payload.get("haa_low"), expected_low)
            and torch.equal(payload.get("haa_high"), expected_high)
        ):
            raise ValueError(
                f"{path} uses the legacy HAA domain; regenerate labels and retrain "
                "with train_api_haa_range.py for the [-1.5, 1.5] running ranges"
            )
        self.network = ApiHaaRangeNetwork(tuple(payload["hidden_dims"])).to(self.device)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.condition_low = self.network.condition_low.detach().clone()
        self.condition_high = self.network.condition_high.detach().clone()
        bank = payload.get("valid_conditions")
        if not isinstance(bank, torch.Tensor) or bank.ndim != 2 or bank.shape[1] != len(self.condition_names) or bank.shape[0] == 0:
            raise ValueError(
                f"{path} has no valid-envelope bank; regenerate labels and retrain so "
                "RL never samples a rejection-range 'none' envelope"
            )
        if torch.any(bank < self.condition_low.cpu()) or torch.any(bank > self.condition_high.cpu()):
            raise ValueError(f"{path} contains an out-of-domain valid-envelope bank entry")
        self.valid_conditions = bank.to(device=self.device, dtype=torch.float)
        self.conditions = torch.empty(num_envs, 5, device=self.device)

    def get_conditions(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.conditions if env_ids is None else self.conditions[env_ids]

    @torch.no_grad()
    def set_conditions(self, values: torch.Tensor, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply explicit five-parameter envelopes through the range network."""
        target_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None else env_ids.to(device=self.device, dtype=torch.long)
        )
        values = torch.as_tensor(values, dtype=torch.float, device=self.device)
        expected = (target_ids.numel(), 5)
        if tuple(values.shape) != expected:
            raise ValueError(f"Expected envelope conditions {expected}, got {tuple(values.shape)}")
        if torch.any(values < self.condition_low) or torch.any(values > self.condition_high):
            raise ValueError("Envelope conditions fall outside the range-network input domain")
        ranges = self.network(values)
        if not torch.all(self._network_range_is_valid(ranges)):
            raise ValueError("HAA range network returned an invalid interval")
        self.conditions[target_ids] = values
        self.ranges[target_ids] = ranges
        return self.ranges[target_ids]

    @torch.no_grad()
    def sample(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not env_ids.numel():
            return self.ranges[env_ids]
        remaining = env_ids
        # ``valid_conditions`` contains only five-vectors for which the exact
        # rejection-range algorithm found all six coupled running intervals.
        # Use it instead of uniform sampling over the whole box, where ``none``
        # has no valid regression target.
        for _ in range(self.max_attempts):
            if not remaining.numel():
                break
            indices = torch.randint(self.valid_conditions.shape[0], (remaining.numel(),), device=self.device)
            condition = self.valid_conditions[indices]
            ranges = self.network(condition)
            valid = self._network_range_is_valid(ranges)
            if valid.any():
                accepted = remaining[valid]
                self.ranges[accepted], self.conditions[accepted] = ranges[valid], condition[valid]
            remaining = remaining[~valid]
        if remaining.numel():
            raise RuntimeError(
                "HAA range network did not produce a >= 0.2-rad running range within "
                f"{self.max_attempts} random envelope samples."
            )
        return self.ranges[env_ids]

    def _network_range_is_valid(self, ranges: torch.Tensor) -> torch.Tensor:
        wide_enough = ((ranges[..., 1] - ranges[..., 0]) >= self.min_width - 1e-6).all(dim=1)
        return torch.isfinite(ranges).all(dim=2).all(dim=1) & wide_enough
