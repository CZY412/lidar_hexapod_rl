"""Runtime loader for the fixed-spider API HAA range network.

The checkpoint is intentionally self describing: it carries the five input
names, six leg names, input bounds, and output bounds used during training.
This module validates those contracts before putting the network in service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


class ApiHaaRangeNetwork(nn.Module):
    """Five envelope values to six ordered ``[lower, upper]`` HAA ranges."""

    def __init__(self, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers = []
        width = 5
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, int(hidden)), nn.ELU()))
            width = int(hidden)
        self.backbone = nn.Sequential(*layers)
        self.center_head = nn.Linear(width, 6)
        self.half_head = nn.Linear(width, 6)
        self.register_buffer("condition_low", torch.empty(5))
        self.register_buffer("condition_high", torch.empty(5))
        self.register_buffer("haa_low", torch.empty(6))
        self.register_buffer("haa_high", torch.empty(6))

    def forward(self, condition: Tensor) -> Tensor:
        normalized = 2.0 * (condition - self.condition_low) / (
            self.condition_high - self.condition_low
        ) - 1.0
        feature = self.backbone(normalized)
        center = self.haa_low + torch.sigmoid(self.center_head(feature)) * (
            self.haa_high - self.haa_low
        )
        max_half = torch.minimum(center - self.haa_low, self.haa_high - center)
        half = torch.sigmoid(self.half_head(feature)) * max_half
        return torch.stack((center - half, center + half), dim=-1)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        *,
        device: torch.device | str,
        condition_names: Sequence[str],
        condition_low: Tensor,
        condition_high: Tensor,
    ) -> "ApiHaaRangeNetwork":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing envelop_3 HAA range network: {path}")
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # PyTorch < 2.0 has no weights_only argument.
            checkpoint = torch.load(path, map_location=device)

        required = {
            "state_dict", "hidden_dims", "condition_names", "leg_names",
            "output_names", "condition_low", "condition_high", "haa_low", "haa_high",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"Invalid envelop_3 HAA range checkpoint; missing {sorted(missing)}")
        checkpoint_names = tuple(checkpoint["condition_names"])
        if checkpoint_names != tuple(condition_names):
            raise ValueError(
                "HAA network input order does not match envelop_3: "
                f"{checkpoint_names} != {tuple(condition_names)}"
            )
        if tuple(checkpoint["output_names"]) != ("lower", "upper"):
            raise ValueError("HAA network must emit [lower, upper] intervals")
        checkpoint_low = torch.as_tensor(checkpoint["condition_low"], dtype=torch.float32, device=device)
        checkpoint_high = torch.as_tensor(checkpoint["condition_high"], dtype=torch.float32, device=device)
        if not torch.allclose(checkpoint_low, condition_low.detach().to(device=device)) or not torch.allclose(
            checkpoint_high, condition_high.detach().to(device=device)
        ):
            raise ValueError("HAA network input bounds do not match envelop_3.envelope.ranges")

        network = cls(checkpoint["hidden_dims"]).to(device)
        network.load_state_dict(checkpoint["state_dict"])
        network.leg_names = tuple(checkpoint["leg_names"])
        if len(network.leg_names) != 6 or len(set(network.leg_names)) != 6:
            raise ValueError(f"Invalid HAA network leg order: {network.leg_names}")
        network.eval()
        return network
