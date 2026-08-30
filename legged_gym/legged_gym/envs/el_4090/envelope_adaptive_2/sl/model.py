"""Policy network for the EA2 supervised-learning pipeline (B1).

The architecture is deliberately **isomorphic to rsl_rl's actor half** of
:class:`~rsl_rl.modules.ActorCriticRecurrent`, so that trained weights can be
transplanted into a PPO policy without any reshaping (see ``export.py``).

    EnvelopeNet                     ActorCriticRecurrent (actor side)
    --------------------------      --------------------------------
    gru.weight_ih_l0                memory_a.rnn.weight_ih_l0
    gru.weight_hh_l0                memory_a.rnn.weight_hh_l0
    gru.bias_ih_l0                  memory_a.rnn.bias_ih_l0
    gru.bias_hh_l0                  memory_a.rnn.bias_hh_l0
    mlp.0.weight / bias             actor.0.weight / bias
    mlp.2.weight / bias             actor.2.weight / bias
    mlp.4.weight / bias             actor.4.weight / bias

Conventions
-----------
* ``nn.GRU`` uses ``batch_first=False``: tensors are ``(seq, batch, dim)``.
  This matches rsl_rl's ``Memory``, which feeds ``(1, num_envs, obs_dim)``.
* The MLP uses the ``Linear -> ELU -> Linear -> ELU -> Linear`` layout that
  rsl_rl builds from ``actor_hidden_dims=[256, 128]``, giving layer indices
  0, 2, 4.
* ``step()`` runs a single frame while carrying the hidden state, which is how
  the policy is used at deployment (and is measurably better than re-running a
  full window each step).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .sl_config import SLModelConfig


def _make_activation(name: str) -> nn.Module:
    key = (name or "elu").lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "lrelu" or key == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"unsupported activation: {name!r}")


class EnvelopeNet(nn.Module):
    """GRU + MLP mapping a 190-D observation sequence to 5 envelope params."""

    def __init__(self, cfg: Optional[SLModelConfig] = None, **overrides):
        super().__init__()
        cfg = cfg or SLModelConfig()
        self.cfg = cfg

        rnn_type = (cfg.rnn_type or "gru").lower()
        if rnn_type != "gru":
            raise ValueError(f"only 'gru' is supported for export parity, got {rnn_type!r}")

        self.gru = nn.GRU(
            input_size=cfg.obs_dim,
            hidden_size=cfg.rnn_hidden_dim,
            num_layers=cfg.rnn_num_layers,
            batch_first=False,  # (seq, batch, dim) -- matches rsl_rl
        )

        dims: List[int] = [cfg.rnn_hidden_dim, *list(cfg.actor_hidden_dims), cfg.action_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_make_activation(cfg.activation))
        self.mlp = nn.Sequential(*layers)

    # -- forward -----------------------------------------------------------
    def forward(self, obs: torch.Tensor, hidden=None) -> torch.Tensor:
        """Full-sequence forward.

        Args:
            obs: ``(seq, batch, 190)``
        Returns:
            ``(seq, batch, 5)``
        """
        out, _ = self.gru(obs, hidden)
        return self.mlp(out)

    def step(self, obs: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-frame forward used for deployment.

        Args:
            obs: ``(1, batch, 190)`` -- a single frame with a leading time axis.
            hidden: previous hidden state or ``None``.
        Returns:
            ``(pred, new_hidden)`` where ``pred`` is ``(batch, 5)``.
        """
        if obs.dim() != 3 or obs.shape[0] != 1:
            raise ValueError(f"step expects (1, batch, obs_dim), got {tuple(obs.shape)}")
        out, new_hidden = self.gru(obs, hidden)
        return self.mlp(out[-1]), new_hidden

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(
            self.cfg.rnn_num_layers, batch_size, self.cfg.rnn_hidden_dim, device=device
        )

    # -- export helpers ----------------------------------------------------
    def actor_state_dict_keys(self) -> Dict[str, str]:
        """Map ``self.state_dict()`` keys onto ``ActorCriticRecurrent`` keys."""
        return {
            "gru.weight_ih_l0": "memory_a.rnn.weight_ih_l0",
            "gru.weight_hh_l0": "memory_a.rnn.weight_hh_l0",
            "gru.bias_ih_l0": "memory_a.rnn.bias_ih_l0",
            "gru.bias_hh_l0": "memory_a.rnn.bias_hh_l0",
            "mlp.0.weight": "actor.0.weight",
            "mlp.0.bias": "actor.0.bias",
            "mlp.2.weight": "actor.2.weight",
            "mlp.2.bias": "actor.2.bias",
            "mlp.4.weight": "actor.4.weight",
            "mlp.4.bias": "actor.4.bias",
        }

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
