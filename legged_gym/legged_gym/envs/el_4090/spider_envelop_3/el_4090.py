"""EL4090 envelope task using the occupied-body API for HAA ranges."""

from __future__ import annotations

import numpy as np
import torch
from isaacgym import gymtorch
from typing import Optional

from legged_gym.envs.el_4090.spider_envelop.el_4090 import EL_4090_ENVELOP
from legged_gym.envs.el_4090.spider_envelop_3.el4090_spider_config import El4090Envelop3Cfg
from legged_gym.envs.el_4090.spider_envelop_3.modal_envelope_range import (
    ModalEnvelopeHaaRangeEstimator,
    NoFeasibleHaaEnvelopeError,
)
from legged_gym.envs.el_4090.spider_envelop_3.allowed_envelope_state import (
    AllowedEnvelopeState,
)


class EL_4090_ENVELOP_3(EL_4090_ENVELOP):
    """Self-contained envelop_3 behavior with API-derived HAA ranges."""

    cfg: El4090Envelop3Cfg

    def __init__(self, cfg: El4090Envelop3Cfg, sim_params, physics_engine, sim_device, headless, task_name="el4090_envelop_3"):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, task_name)

    def _init_buffers(self) -> None:
        # Initialize envelop_3's five-value external state and API estimator.
        EL_4090_ENVELOP._init_buffers(self)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.action_history_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False,
        )
        self.envelope_state = AllowedEnvelopeState(
            self.cfg.envelope, self.num_envs, self.device,
        )
        self.condition_names = list(self.envelope_state.condition_names)
        self.condition_dim = self.envelope_state.condition_dim
        self.condition_low = self.envelope_state.low
        self.condition_high = self.envelope_state.high
        haa_leg_names = tuple(
            self.dof_names[index].split("_", 1)[0] for index in self.haa_indices.tolist()
        )
        self._haa_range_output_indices = torch.arange(
            len(haa_leg_names), dtype=torch.long, device=self.device,
        )
        self.haa_range_estimator = self._make_haa_range_estimator(None)
        self.haa_swing_ranges = torch.zeros(
            self.num_envs, len(haa_leg_names), 2, dtype=torch.float,
            device=self.device, requires_grad=False,
        )
        # These are derived from the API range centers, not sampled inputs.
        self.api_morphology_priors = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False,
        )
        self._haa_range_condition_cache = torch.full(
            (self.num_envs, self.condition_dim), float("nan"), dtype=torch.float,
            device=self.device, requires_grad=False,
        )
        self._refresh_haa_swing_ranges()

    def _make_haa_range_estimator(self, _unused_config):
        return ModalEnvelopeHaaRangeEstimator(
            self.cfg.envelope_api_range,
            condition_names=self.condition_names,
            leg_names=tuple(self.dof_names[index].split("_", 1)[0] for index in self.haa_indices.tolist()),
            dof_names=self.dof_names,
            spider_dof_pos=self.default_dof_pos[0],
            mammal_dof_pos=self.mammal_default_dof_pos[0],
        )

    @torch.no_grad()
    def _refresh_haa_swing_ranges(self, env_ids=None) -> None:
        if env_ids is None:
            condition = self._get_structure_condition()
            changed = torch.any(condition != self._haa_range_condition_cache, dim=1)
            env_ids = changed.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        condition = self._get_structure_condition()[env_ids]
        ranges, priors = self.haa_range_estimator.compute(condition)
        if ranges.shape != self.haa_swing_ranges[env_ids].shape:
            raise RuntimeError(
                f"HAA estimator returned {tuple(ranges.shape)}, expected "
                f"{tuple(self.haa_swing_ranges[env_ids].shape)}"
            )
        self.haa_swing_ranges[env_ids] = ranges
        self.api_morphology_priors[env_ids] = priors
        self._haa_range_condition_cache[env_ids] = condition

    def _get_morphology_prior(self, prior_name):
        columns = {
            "morphology_front_prior": 0,
            "morphology_middle_prior": 1,
            "morphology_back_prior": 2,
        }
        column = columns.get(prior_name)
        if column is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        return self.api_morphology_priors[:, column]

    def _get_body_height_morphology_prior(self):
        """Keep height/reward interpolation aligned with the derived modal pose."""
        return self.api_morphology_priors.mean(dim=1)

    def step(self, actions):
        previous_actions = self.last_actions.clone()
        result = super().step(actions)
        self._update_action_history(previous_actions, result[3])
        return result

    @torch.no_grad()
    def _update_action_history(self, previous_actions: torch.Tensor, dones: torch.Tensor) -> None:
        done_mask = dones.to(dtype=torch.bool)
        self.last_last_actions[:] = previous_actions
        self.action_history_steps[~done_mask] = torch.clamp(
            self.action_history_steps[~done_mask] + 1, max=2,
        )
        self.last_last_actions[done_mask] = 0.0
        self.last_actions[done_mask] = 0.0
        self.action_history_steps[done_mask] = 0

    def _get_noise_scale_vec(self, cfg) -> torch.Tensor:
        """Noise vector for the explicit 66-D proprioceptive observation."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0
        noise_vec[12:30] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[30:48] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[48:66] = 0.0
        return noise_vec

    def compute_observations(self) -> None:
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, 0:1] * self.command_lin_vel_scale,
            self.commands[:, 1:2] * self.command_lin_vel_scale,
            self.commands[:, 2:3] * self.command_ang_vel_scale,
            (self.dof_pos - self.embedded_state_default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        if getattr(self.cfg.lidar, "enable", False):
            self.obs_buf = torch.cat((self.obs_buf, self._get_lidar_observations()), dim=-1)
        elif self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.45 - self.measured_heights, -5, 5.0)
            self.obs_buf = torch.cat((self.obs_buf, heights * self.obs_scales.height_measurements), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_structure_condition(self) -> torch.Tensor:
        return self.envelope_state.get()

    def _get_haa_range_center_and_half(self):
        lower, upper = self.haa_swing_ranges[..., 0], self.haa_swing_ranges[..., 1]
        return 0.5 * (lower + upper), 0.5 * (upper - lower)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        for command_index, range_name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            low, high = self.command_ranges[range_name]
            self.commands[env_ids, command_index] = low + torch.rand(
                env_ids.numel(), dtype=torch.float, device=self.device,
            ) * (high - low)
        self._sample_valid_envelope_conditions(env_ids)
        if getattr(self.cfg.rewards, "reset_structure_transition_on_resample", True):
            self.embedded_state_transition_time[env_ids] = 0.0

    @torch.no_grad()
    def _sample_valid_envelope_conditions(self, env_ids: torch.Tensor) -> None:
        """Rejection-sample geometry that admits a pose in all modal HAA domains."""
        max_attempts = int(getattr(self.cfg.envelope_api_range, "sampling_max_attempts", 32))
        pending = env_ids.clone()
        last_error = None
        for _ in range(max_attempts):
            self.envelope_state.sample(pending)
            accepted = []
            for env_id in pending.tolist():
                condition = self._get_structure_condition()[env_id:env_id + 1]
                try:
                    ranges, priors = self.haa_range_estimator.compute(condition)
                except NoFeasibleHaaEnvelopeError as error:
                    last_error = error
                    continue
                self.haa_swing_ranges[env_id] = ranges[0]
                self.api_morphology_priors[env_id] = priors[0]
                self._haa_range_condition_cache[env_id] = condition[0]
                accepted.append(env_id)
            if not accepted:
                continue
            accepted_ids = torch.tensor(accepted, dtype=torch.long, device=self.device)
            keep = ~torch.isin(pending, accepted_ids)
            pending = pending[keep]
            if pending.numel() == 0:
                return
        raise RuntimeError(
            f"Could not sample a feasible envelop_3 geometry after {max_attempts} attempts. "
            "Widen envelope.ranges or increase envelope_api_range.margin. "
            f"Last API result: {last_error}"
        )

    @torch.no_grad()
    def set_envelope_condition(self, values, env_ids: Optional[torch.Tensor] = None, *,
                               derive_priors: bool = True) -> torch.Tensor:
        target_ids = (torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                      if env_ids is None else env_ids.to(device=self.device, dtype=torch.long))
        previous = self.envelope_state.get(target_ids).clone()
        updated = self.envelope_state.set(values, env_ids, derive_priors=derive_priors)
        try:
            self._refresh_haa_swing_ranges(target_ids)
        except NoFeasibleHaaEnvelopeError:
            self.envelope_state.set(previous, target_ids)
            raise
        self.embedded_state_default_dof_pos[target_ids] = self._get_condition_target_dof_pos()[target_ids]
        return updated

    def _post_physics_step_callback(self) -> None:
        super()._post_physics_step_callback()
        self._refresh_haa_swing_ranges()

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.control.control_type != "P":
            return super()._compute_torques(actions)
        target = actions * self.cfg.control.action_scale + self.embedded_state_default_dof_pos
        torques = self.p_gains * (target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.last_last_actions[env_ids] = 0.0
        self.action_history_steps[env_ids] = 0
        target = self._get_condition_target_dof_pos()[env_ids]
        self.embedded_state_default_dof_pos[env_ids] = target
        self.filtered_embedded_state_default_dof_pos[env_ids] = target
        self.dof_pos[env_ids] = target
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32),
        )

    def _reward_haa_range_violation(self) -> torch.Tensor:
        margin = float(getattr(self.cfg.rewards, "haa_range_margin", 0.0))
        lower, upper = self.haa_swing_ranges[..., 0] - margin, self.haa_swing_ranges[..., 1] + margin
        violation = torch.relu(lower - self.dof_pos[:, self.haa_indices]) + torch.relu(self.dof_pos[:, self.haa_indices] - upper)
        return torch.mean(torch.square(violation / (0.5 * (upper - lower)).clamp_min(1e-6)), dim=1)

    def _reward_action_jerk(self) -> torch.Tensor:
        second_difference = self.actions - 2.0 * self.last_actions + self.last_last_actions
        return torch.sum(torch.square(second_difference), dim=1) * (self.action_history_steps >= 2).to(self.actions.dtype)

    def _spider_morphology_weight(self, prior_name: Optional[str] = None) -> torch.Tensor:
        if prior_name is not None:
            return 1.0 - self._get_morphology_prior(prior_name)
        priors = torch.stack([self._get_morphology_prior(name) for name in (
            "morphology_front_prior", "morphology_middle_prior", "morphology_back_prior",
        )], dim=1)
        return 1.0 - torch.max(priors, dim=1).values

    def _reward_stand_contact(self) -> torch.Tensor:
        threshold = float(getattr(self.cfg.rewards, "stand_contact_threshold", 1.0))
        command_threshold = float(getattr(self.cfg.rewards, "stand_command_threshold", self.speed_min))
        contact = self.contact_forces[:, self.feet_indices, 2] > threshold
        standing = torch.linalg.vector_norm(self.commands[:, :3], dim=1) < command_threshold
        return torch.sum((~contact).to(self.dof_pos.dtype), dim=1) * standing * self._get_gait_reward_active_mask()

    def _reward_spider_front_haa_posture(self) -> torch.Tensor:
        indices = self.morphology_prior_haa_indices["morphology_front_prior"]
        excess = torch.relu(torch.abs(self.dof_pos[:, indices] - self.embedded_state_default_dof_pos[:, indices]) - float(
            getattr(self.cfg.rewards, "spider_front_haa_free_error", 0.50)
        ))
        moving = (torch.linalg.vector_norm(self.commands[:, :3], dim=1) > self.speed_min).to(self.dof_pos.dtype)
        return torch.mean(torch.square(excess), dim=1) * torch.square(
            self._spider_morphology_weight("morphology_front_prior")
        ) * moving

    def _reward_spider_straightness(self) -> torch.Tensor:
        min_forward = float(getattr(self.cfg.rewards, "spider_straight_min_forward_command", 0.15))
        tolerance = float(getattr(self.cfg.rewards, "spider_straight_command_tolerance", 0.08))
        yaw_weight = float(getattr(self.cfg.rewards, "spider_straight_yaw_weight", 0.50))
        straight = ((torch.abs(self.commands[:, 0]) > min_forward)
                    & (torch.abs(self.commands[:, 1]) < tolerance)
                    & (torch.abs(self.commands[:, 2]) < tolerance)).to(self.dof_pos.dtype)
        drift_and_yaw = torch.square(self.base_lin_vel[:, 1]) + yaw_weight * torch.square(self.base_ang_vel[:, 2])
        return drift_and_yaw * torch.square(self._spider_morphology_weight()) * straight

    def _reward_base_height(self) -> torch.Tensor:
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.abs(base_height - self._get_base_height_target())

    def _draw_envelope_condition_vis(self):
        """Draw pure API allowed and occupied support envelopes."""
        if getattr(self, "viewer", None) is None:
            return
        from el4090_envelope.geometry import support_polygon
        from legged_gym.utils.math_utils import quat_apply_yaw

        env_ids = getattr(self.cfg.commands, "envelope_debug_env_ids", [0])
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        for env_id in env_ids:
            if env_id < 0 or env_id >= self.num_envs:
                continue
            occupied_support = self.haa_range_estimator.occupied_support_from_isaac_dof(
                self.dof_pos[env_id], self.dof_names,
            )
            allowed_support = self.haa_range_estimator.allowed_support(
                self._get_structure_condition()[env_id].detach().cpu(),
            )
            for support, color, height in (
                (allowed_support, (0.0, 0.85, 1.0), 0.025),
                (occupied_support, (1.0, 0.45, 0.0), 0.035),
            ):
                polygon = support_polygon(self.haa_range_estimator.directions, support)
                local_points = self.dof_pos.new_tensor(
                    np.column_stack((polygon, np.zeros(len(polygon)))),
                )
                translation = self.root_states[env_id, :3].clone()
                translation[2] = self.env_origins[env_id, 2] + height
                world_points = quat_apply_yaw(
                    self.base_quat[env_id].repeat(local_points.shape[0], 1), local_points,
                ) + translation.unsqueeze(0)
                points = world_points.detach().cpu().numpy().astype(np.float32)
                vertices = np.stack((points, np.roll(points, -1, axis=0)), axis=1)
                colors = np.tile(np.array(color, dtype=np.float32), (len(points), 1))
                self.gym.add_lines(self.viewer, self.envs[env_id], len(points), vertices, colors)
