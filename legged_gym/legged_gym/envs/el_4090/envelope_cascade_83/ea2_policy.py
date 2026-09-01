"""EA2 SL-exported GRU policy loader for the cascade demo.

The pinned checkpoint (``checkpoints/ea2_envelope.pt``) is the rsl_rl
checkpoint produced by ``sl.scripts.export``: the SL network's s∈[0,1]
output has been folded into the actor's last layer, so the policy consumes
190-dim observations and emits **raw actions** that the
:class:`~envelope_bridge.EnvelopeBridge` maps to envelope parameters.

``make_alg_runner`` cannot build this policy against the cascade env (the
runner derives observation dims from ``env.num_obs`` = 83), so the
``ActorCriticRecurrent`` is constructed explicitly here with the exact
architecture recorded in the live EA2 PPO config, and the state dict is
loaded strictly.  GRU hidden-state management mirrors the deployment mode
validated in EA2 closed-loop eval (``play_ea2.py``): stateful per-step
inference with ``reset(dones)`` zeroing finished episodes.
"""

from __future__ import annotations

from pathlib import Path

import torch
from rsl_rl.modules import ActorCriticRecurrent

from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
    El4090EA2Cfg,
    El4090EA2CfgPPO,
)

_RSL_REQUIRED_KEYS = ("model_state_dict", "optimizer_state_dict", "iter", "infos")


class Ea2Policy:
    """Stateful GRU inference wrapper around the EA2 exported policy."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | str,
        num_observations: int | None = None,
        num_actions: int | None = None,
    ) -> None:
        if num_observations is None:
            num_observations = int(El4090EA2Cfg.env.num_observations)
        if num_actions is None:
            num_actions = int(El4090EA2Cfg.env.num_actions)

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"EA2 cascade checkpoint not found: {checkpoint_path} "
                "(see envelope_cascade/checkpoints/README.md)"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing = [key for key in _RSL_REQUIRED_KEYS if key not in checkpoint]
        if missing:
            raise ValueError(
                f"EA2 checkpoint {checkpoint_path} is missing rsl_rl keys: {missing}"
            )
        state_dict = checkpoint["model_state_dict"]
        if any("normaliz" in key.lower() for key in state_dict):
            raise ValueError(
                "EA2 checkpoint contains normalization state; the cascade "
                "loader does not know how to apply it (see checkpoints/README.md)"
            )

        policy_cfg = El4090EA2CfgPPO.policy
        self.policy = ActorCriticRecurrent(
            num_actor_obs=num_observations,
            num_critic_obs=num_observations,
            num_actions=num_actions,
            actor_hidden_dims=list(policy_cfg.actor_hidden_dims),
            critic_hidden_dims=list(policy_cfg.critic_hidden_dims),
            activation=str(policy_cfg.activation),
            rnn_type=str(policy_cfg.rnn_type),
            rnn_hidden_dim=int(policy_cfg.rnn_hidden_dim),
            rnn_num_layers=int(policy_cfg.rnn_num_layers),
            init_noise_std=float(policy_cfg.init_noise_std),
        )
        self.policy.load_state_dict(state_dict, strict=True)
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()
        self.num_observations = num_observations
        self.num_actions = num_actions

    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        """One stateful inference step for ``(num_envs, num_observations)``."""
        if tuple(observations.shape[-1:]) != (self.num_observations,):
            raise ValueError(
                f"EA2 policy expects {self.num_observations} obs dims, got "
                f"{tuple(observations.shape)}"
            )
        return self.policy.act_inference(observations.to(self.device))

    def reset(self, dones: torch.Tensor) -> None:
        """Zero the GRU hidden state of envs flagged in ``dones``."""
        self.policy.reset(dones.to(self.device))
