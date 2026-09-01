"""EA2 raw action → SE2 envelope condition bridge.

Closes the 5→8 contract between the two trained tasks:

* EA2 policy emits 5 raw actions; ``map_actions_to_params`` (the repo-wide
  single source of truth, imported from the EA2 env module) maps them to
  clamped envelope parameters ``params5`` via ``mid + a * k * (high - low)``
  with ``k = soft_dof_pos_limit / (2 * action_max)``.
* ``envelope_params_to_condition`` derives the 8-dim SE2 condition by
  appending placeholder priors and running the shared
  ``apply_env_morphology_priors`` (idempotent w.r.t. already-derived
  priors, verified by tests).

The bridge is intentionally stateless and torch-only (no Isaac Gym), so the
cascade contract tests can run without a simulation.
"""

from __future__ import annotations

import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    map_actions_to_params,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    envelope_params_to_condition,
)


class EnvelopeBridge:
    """Map EA2 raw actions onto the SE2 envelope condition space."""

    def __init__(
        self,
        envelope_state,
        soft_dof_pos_limit: float,
        action_max: float,
    ) -> None:
        """Bind the bridge to a live SE2 :class:`EnvelopeConditionState`.

        ``soft_dof_pos_limit`` / ``action_max`` must match the values used
        when the EA2 checkpoint was exported (the s→a affine is folded into
        the actor's last layer).  The cascade env asserts this against the
        pinned ``fold_scale`` before constructing the bridge.
        """
        self.envelope_state = envelope_state
        self.spec = envelope_state.spec
        self.low5 = envelope_state.low[:5].clone()
        self.high5 = envelope_state.high[:5].clone()
        self.soft_dof_pos_limit = float(soft_dof_pos_limit)
        self.action_max = float(action_max)
        self.fold_scale = self.soft_dof_pos_limit / (2.0 * self.action_max)

    def params_from_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Return hard-clamped ``params5`` for EA2 raw actions."""
        return map_actions_to_params(
            raw_action,
            self.low5,
            self.high5,
            self.soft_dof_pos_limit,
            self.action_max,
        )

    def condition_from_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Return the derived 8-dim condition for EA2 raw actions."""
        return envelope_params_to_condition(self.params_from_action(raw_action), self.spec)

    def __call__(self, raw_action: torch.Tensor):
        """Return ``(condition8, params5)``; the env feeds ``condition8`` to
        :meth:`set_envelope_condition` and keeps ``params5`` for telemetry."""
        params5 = self.params_from_action(raw_action)
        return envelope_params_to_condition(params5, self.spec), params5
