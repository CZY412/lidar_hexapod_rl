"""EL_4090_CASCADE: SE2 locomotion driven by EA2 point-cloud perception.

Data flow per control step (50 Hz):

1. ``super().post_physics_step()`` — SE2 refreshes buffers, resamples
   commands (on schedule), checks termination, computes rewards, resets
   finished envs and builds the 83-dim SE2 observation.
2. The EA2 perception updates the sensor pose, refreshes the 187-channel
   range image on its 10 Hz clock (first step refreshes; envs reset this
   step keep an empty image until the next scan — EA2's empty-frame
   contract), and assembles the 190-dim observation from *measured*
   body-frame ego motion.
3. The pinned EA2 GRU policy infers 5 raw actions (hidden state reset for
   envs flagged done this step).
4. The bridge maps raw actions → ``params5`` → derived ``condition8``;
   ``set_envelope_condition`` clamps, refreshes the HAA range network and
   the morphology preset; the SE2 observation is then rebuilt so the new
   envelope is visible to the locomotion policy in the same step.

``ea2.enable=False`` restores pure SE2 behaviour (randomly sampled
envelopes) — used as an A/B baseline.
"""

from __future__ import annotations

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.el_4090.envelope_cascade_83.ea2_perception import Ea2Perception
from legged_gym.envs.el_4090.envelope_cascade_83.ea2_policy import Ea2Policy
from legged_gym.envs.el_4090.envelope_cascade_83.el4090_cascade_config import (
    El4090CascadeCfg,
)
from legged_gym.envs.el_4090.envelope_cascade_83.envelope_bridge import EnvelopeBridge
from legged_gym.envs.el_4090.spider_envelop_2.el_4090 import EL_4090_ENVELOP_2

_COMMAND_NAMES = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
_GEOMETRY_NAMES = ("front_width", "middle_width", "back_width", "forward_limit")


class EL_4090_CASCADE(EL_4090_ENVELOP_2):
    """Envelope-cascade EL4090: EA2 perception replaces sampled envelopes."""

    cfg: El4090CascadeCfg

    def __init__(
        self,
        cfg: El4090CascadeCfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
        task_name: str = "el4090_cascade",
    ):
        self._cascade_ready = False
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, task_name)
        if getattr(self.cfg.ea2, "enable", True):
            self._init_cascade()
            self._cascade_ready = True

    # ── setup ───────────────────────────────────────────────────────────

    def _init_cascade(self) -> None:
        ea2 = self.cfg.ea2
        self._cascade_bridge = EnvelopeBridge(
            self.envelope_state, ea2.soft_dof_pos_limit, ea2.action_max
        )
        if abs(self._cascade_bridge.fold_scale - float(ea2.fold_scale)) > 1e-9:
            raise ValueError(
                "EA2 fold-scale drift: live config yields "
                f"{self._cascade_bridge.fold_scale:.6f} but the pinned "
                f"checkpoint was folded with {float(ea2.fold_scale):.6f} "
                "(re-export the EA2 policy or update ea2.fold_scale; see "
                "envelope_cascade_83/checkpoints/README.md)"
            )
        checkpoint = str(ea2.checkpoint).format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self._cascade_policy = Ea2Policy(checkpoint, self.device)
        self._cascade_perception = Ea2Perception(
            num_envs=self.num_envs,
            device=self.device,
            cfg=ea2,
            dt=self.dt,
            terrain_vertices=self.terrain.vertices,
            terrain_triangles=self.terrain.triangles,
            border_size=float(self.cfg.terrain.border_size),
        )
        self._cascade_max_condition = self._build_max_condition()
        self._cascade_mid_condition = (
            0.5 * (self.envelope_state.low + self.envelope_state.high)
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._cascade_params5 = torch.zeros(self.num_envs, 5, device=self.device)
        self._cascade_cond8 = torch.zeros(self.num_envs, 8, device=self.device)
        self._cascade_action = torch.zeros(self.num_envs, 5, device=self.device)
        self._cascade_action_finite = True

    def _build_max_condition(self) -> torch.Tensor:
        """Largest-footprint condition template (priors derived on write)."""
        state = self.envelope_state
        names = list(state.condition_names)
        condition = (0.5 * (state.low + state.high)).unsqueeze(0).repeat(self.num_envs, 1)
        for name in _GEOMETRY_NAMES:
            index = names.index(name)
            condition[:, index] = state.high[index]
        backward = names.index("backward_limit")
        condition[:, backward] = state.low[backward]  # most negative = widest rear
        return condition

    def _cascade_birth_condition(self) -> torch.Tensor:
        mode = str(getattr(self.cfg.ea2, "reset_condition", "max"))
        if mode == "max":
            return self._cascade_max_condition
        if mode == "midpoint":
            return self._cascade_mid_condition
        raise ValueError(f"unknown ea2.reset_condition {mode!r}")

    # ── command / envelope resampling ───────────────────────────────────

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """Locomotion commands as in SE2; envelopes come from the cascade.

        On episode reset the condition is set to the configured birth
        preset (default: maximum envelope) so the robot spawns at a sane
        morphology; the EA2 policy overwrites it in the same control step.
        """
        if env_ids.numel() == 0:
            return
        for command_index, range_name in enumerate(_COMMAND_NAMES):
            low, high = self.command_ranges[range_name]
            self.commands[env_ids, command_index] = low + torch.rand(
                env_ids.numel(), dtype=torch.float, device=self.device
            ) * (high - low)
        if self._cascade_ready:
            self.set_envelope_condition(self._cascade_birth_condition()[env_ids], env_ids)
        else:
            self.envelope_state.sample(env_ids)
        if getattr(self.cfg.rewards, "reset_structure_transition_on_resample", True):
            self.embedded_state_transition_time[env_ids] = 0.0
            self.filtered_embedded_state_default_dof_pos[env_ids] = self.default_dof_pos
        # set_envelope_condition already refreshed the HAA ranges on the
        # cascade path; the cache makes this a no-op there and covers the
        # sampled fallback path (mirrors EL_4090_ENVELOP_2._resample_commands).
        if hasattr(self, "haa_range_estimator"):
            self._refresh_haa_swing_ranges(env_ids)

    # ── per-step cascade bridge ─────────────────────────────────────────

    def post_physics_step(self) -> None:
        super().post_physics_step()
        if not self._cascade_ready:
            return

        perception = self._cascade_perception
        perception.update_pose(self.base_pos, self.base_quat)
        perception.refresh()
        # Mirrors EA2's ordering: the scan runs first, then finished envs are
        # emptied — a reset env keeps the empty frame until the next scan.
        perception.mark_stale(self.reset_buf.nonzero(as_tuple=False).flatten())

        ego = perception.ego_motion_from(self.base_lin_vel, self.base_ang_vel)
        obs190 = perception.observe(ego)
        raw_action = self._cascade_policy.act(obs190)
        self._cascade_policy.reset(self.reset_buf)

        condition8, params5 = self._cascade_bridge(raw_action)
        # set_envelope_condition refreshes the HAA ranges and the morphology
        # preset (embedded_state_default_dof_pos) in-place.
        self.set_envelope_condition(condition8)
        self._cascade_action.copy_(raw_action)
        self._cascade_params5.copy_(params5)
        self._cascade_cond8.copy_(condition8)
        self._cascade_action_finite = bool(torch.isfinite(raw_action).all())

        # rebuild the SE2 observation so the fresh envelope is visible now
        self.compute_observations()

    # ── diagnostics (play / tests) ──────────────────────────────────────

    @property
    def cascade_params5(self) -> torch.Tensor:
        return self._cascade_params5

    @property
    def cascade_condition8(self) -> torch.Tensor:
        return self._cascade_cond8

    @property
    def cascade_action_finite(self) -> bool:
        return self._cascade_action_finite

    def cascade_debug_summary(self) -> dict:
        """Cheap per-call telemetry; safe to call every print interval."""
        bridge = self._cascade_bridge
        span = (bridge.high5 - bridge.low5).clamp_min(1e-6)
        at_bound = ((self._cascade_params5 - bridge.low5).abs() < 1e-4) | (
            (bridge.high5 - self._cascade_params5).abs() < 1e-4
        )
        return {
            "params5": self._cascade_params5[0].detach().cpu().tolist(),
            "refresh_count": self._cascade_perception.refresh_count,
            "stale_envs": int(self._cascade_perception.stale.sum().item()),
            "action_finite": self._cascade_action_finite,
            "bound_ratio": float(at_bound.float().mean().item()),
        }

    def cascade_debug_cloud(self):
        """Latest noisy 187-point body-frame cloud for viewer drawing."""
        return self._cascade_perception.debug_cloud()
