"""EL4090 task with directly sampled HAA swing ranges."""

from __future__ import annotations

import torch
from isaacgym import gymtorch
from typing import Optional

from legged_gym.envs.el_4090.spider_envelop.el_4090 import EL_4090_ENVELOP
from legged_gym.envs.el_4090.spider_envelop_3.el4090_spider_config import El4090Envelop3Cfg
from legged_gym.envs.el_4090.spider_envelop_3.haa_range_state import HaaRangeState


class EL_4090_ENVELOP_3(EL_4090_ENVELOP):
    """Self-contained envelop_3 behavior with directly sampled HAA ranges."""

    cfg: El4090Envelop3Cfg

    def __init__(self, cfg: El4090Envelop3Cfg, sim_params, physics_engine, sim_device, headless, task_name="el4090_envelop_3"):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, task_name)

    def _init_buffers(self) -> None:
        # HAA intervals are sampled directly; no envelope state or model is used.
        EL_4090_ENVELOP._init_buffers(self)
        # The parent task allocates this legacy morphology buffer. Envelop_3
        # replaces it with an HAA interval-centre bias, without changing the
        # unique static ``default_dof_pos`` buffer.
        del self.mammal_default_dof_pos
        self.last_last_actions = torch.zeros_like(self.actions)
        self.action_history_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False,
        )
        haa_leg_names = tuple(
            self.dof_names[index].split("_", 1)[0] for index in self.haa_indices.tolist()
        )
        # Sample ranges directly. ``haa_interference.py`` remains a standalone
        # diagnostic because checking all environments' capsule corners can be
        # prohibitively expensive on the training GPU.
        self.haa_range_state = HaaRangeState(self.cfg.haa_range, self.num_envs, self.device)
        if haa_leg_names != self.haa_range_state.leg_names:
            raise ValueError(f"Configured HAA leg order {self.haa_range_state.leg_names} != Isaac order {haa_leg_names}")
        self.haa_swing_ranges = self.haa_range_state.get()
        self.haa_range_bias = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.haa_range_state.sample(all_env_ids)
        self._refresh_haa_range_bias(all_env_ids)
        self.embedded_state_default_dof_pos[:] = self._get_condition_target_dof_pos()
        self.filtered_embedded_state_default_dof_pos[:] = self.embedded_state_default_dof_pos

    def _get_condition_target_dof_pos(self) -> torch.Tensor:
        """Return the static spider default plus the sampled HAA-centre bias."""
        target = self.default_dof_pos.repeat(self.num_envs, 1)
        if hasattr(self, "haa_range_bias"):
            target += self.haa_range_bias
        return target

    @torch.no_grad()
    def _refresh_haa_range_bias(self, env_ids: torch.Tensor) -> None:
        """Use each sampled HAA interval centre as an additive posture bias."""
        if env_ids.numel() == 0:
            return
        centers, _ = self._get_haa_range_center_and_half()
        scale = float(getattr(self.cfg.haa_range, "center_bias_scale", 1.0))
        self.haa_range_bias[env_ids] = 0.0
        self.haa_range_bias[env_ids[:, None], self.haa_indices] = centers[env_ids] * scale

    def _get_base_height_target(self) -> torch.Tensor:
        return torch.full(
            (self.num_envs,), float(self.cfg.rewards.target_height), device=self.device,
        )

    def _get_gait_reward_active_mask(self) -> torch.Tensor:
        """Keep gait rewards active immediately; envelop_3 has no posture transition."""
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

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
        """Noise vector for proprioception, target height, and HAA ranges."""
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
        noise_vec[48:67] = 0.0
        return noise_vec

    def compute_observations(self) -> None:
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, 0:1] * self.command_lin_vel_scale,
            self.commands[:, 1:2] * self.command_lin_vel_scale,
            self.commands[:, 2:3] * self.command_ang_vel_scale,
            (self.dof_pos - self._get_condition_target_dof_pos()) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self._get_base_height_target().unsqueeze(1) * self.obs_scales.height_measurements,
            self.haa_swing_ranges.reshape(self.num_envs, -1),
        ), dim=-1)
        if getattr(self.cfg.lidar, "enable", False):
            self.obs_buf = torch.cat((self.obs_buf, self._get_lidar_observations()), dim=-1)
        elif self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.45 - self.measured_heights, -5, 5.0)
            self.obs_buf = torch.cat((self.obs_buf, heights * self.obs_scales.height_measurements), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

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
        self.haa_range_state.sample(env_ids)
        self._refresh_haa_range_bias(env_ids)

    @torch.no_grad()
    def set_haa_ranges(self, values, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Set direct HAA intervals for playback or evaluation."""
        target_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None else env_ids.to(device=self.device, dtype=torch.long)
        )
        ranges = self.haa_range_state.set(values, target_ids)
        self._refresh_haa_range_bias(target_ids)
        self.embedded_state_default_dof_pos[target_ids] = self._get_condition_target_dof_pos()[target_ids]
        return ranges

    @torch.no_grad()
    def sample_haa_ranges(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Directly resample valid-width, left/right-coupled HAA intervals."""
        target_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None else env_ids.to(device=self.device, dtype=torch.long)
        )
        ranges = self.haa_range_state.sample(target_ids)
        self._refresh_haa_range_bias(target_ids)
        self.embedded_state_default_dof_pos[target_ids] = self._get_condition_target_dof_pos()[target_ids]
        return ranges

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.control.control_type != "P":
            return super()._compute_torques(actions)
        target = actions * self.cfg.control.action_scale + self._get_condition_target_dof_pos()
        torques = self.p_gains * (target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.last_last_actions[env_ids] = 0.0
        self.action_history_steps[env_ids] = 0
        target = self._get_condition_target_dof_pos()[env_ids]
        self.dof_pos[env_ids] = target
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32),
        )

    def _reward_haa_range_violation(self) -> torch.Tensor:
        """Gently discourage leaving the sampled HAA interval.

        The old clipped quadratic saturated for a large initial violation, so
        it supplied no distinction between "slightly outside" and "far
        outside".  This Huber penalty remains smooth at the boundary and is
        annealed from a small weight, allowing early walking exploration.
        """
        lower, upper = self.haa_swing_ranges[..., 0], self.haa_swing_ranges[..., 1]
        outside = torch.relu(lower - self.dof_pos[:, self.haa_indices])
        outside += torch.relu(self.dof_pos[:, self.haa_indices] - upper)
        softness = max(float(getattr(self.cfg.rewards, "haa_range_violation_softness", 0.25)), 1e-6)
        normalized = outside / softness
        huber = torch.where(
            normalized <= 1.0,
            0.5 * torch.square(normalized),
            normalized - 0.5,
        )
        initial = float(getattr(self.cfg.rewards, "haa_range_violation_initial_weight", 0.10))
        warmup_steps = max(int(getattr(self.cfg.rewards, "haa_range_violation_warmup_steps", 1)), 1)
        progress = min(float(self.common_step_counter) / warmup_steps, 1.0)
        curriculum_weight = initial + (1.0 - initial) * progress
        return torch.mean(huber, dim=1) * curriculum_weight

    def _reward_haa_range_compliance(self) -> torch.Tensor:
        """Reward being anywhere in the safe interior, without a pose target."""
        lower, upper = self.haa_swing_ranges[..., 0], self.haa_swing_ranges[..., 1]
        margin = float(getattr(self.cfg.rewards, "haa_range_compliance_margin", 0.04))
        # The sampled minimum width is 0.3 rad, so this remains a non-empty
        # interior with the default 0.04 rad margin.
        interior_lower, interior_upper = lower + margin, upper - margin
        signed_interior_distance = torch.minimum(
            self.dof_pos[:, self.haa_indices] - interior_lower,
            interior_upper - self.dof_pos[:, self.haa_indices],
        )
        softness = max(float(getattr(self.cfg.rewards, "haa_range_compliance_softness", 0.04)), 1e-6)
        return torch.mean(torch.sigmoid(signed_interior_distance / softness), dim=1)

    def _reward_haa_range_swing(self) -> torch.Tensor:
        """Reward useful HAA motion while withholding reward near range edges.

        Motion has to reach ``target_excursion_ratio`` of the available
        half-range to receive its full excursion factor. The final
        ``boundary_buffer_ratio`` of either side receives no swing reward, so
        rapidly rubbing an interval edge cannot be a rewarding strategy.
        """
        lower, upper = self.haa_swing_ranges[..., 0], self.haa_swing_ranges[..., 1]
        center = 0.5 * (lower + upper)
        half_range = (0.5 * (upper - lower)).clamp_min(1e-6)
        normalized_excursion = torch.abs(self.dof_pos[:, self.haa_indices] - center) / half_range

        target_excursion = float(getattr(
            self.cfg.rewards, "haa_range_swing_target_excursion_ratio", 0.65,
        ))
        boundary_buffer = float(getattr(
            self.cfg.rewards, "haa_range_swing_boundary_buffer_ratio", 0.12,
        ))
        target_excursion = min(max(target_excursion, 1e-6), 1.0)
        boundary_buffer = min(max(boundary_buffer, 1e-6), 1.0 - 1e-6)
        excursion_factor = torch.clamp(normalized_excursion / target_excursion, max=1.0)
        interior_factor = torch.clamp(
            (1.0 - normalized_excursion) / boundary_buffer,
            min=0.0,
            max=1.0,
        )
        velocity_clip = max(float(getattr(
            self.cfg.rewards, "haa_range_swing_velocity_clip", 4.0,
        )), 1e-6)
        velocity_factor = torch.clamp(
            torch.abs(self.dof_vel[:, self.haa_indices]) / velocity_clip,
            max=1.0,
        )
        min_command = float(getattr(self.cfg.rewards, "haa_range_swing_min_command", 0.10))
        moving = (torch.linalg.vector_norm(self.commands[:, :3], dim=1) > min_command).to(self.dof_pos.dtype)
        return torch.mean(velocity_factor * excursion_factor * interior_factor, dim=1) * moving

    def _reward_dof_pos_limits(self) -> torch.Tensor:
        """Keep generic joint-limit protection for HFE/KFE, not HAA ranges.

        HAA's only task-specific position constraint is now
        ``_reward_haa_range_violation`` above.
        """
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        out_of_limits[:, self.haa_indices] = 0.0
        return torch.sum(out_of_limits, dim=1)

    def _reward_stand_still(self) -> torch.Tensor:
        """At rest, hold the static spider posture plus the HAA-centre bias."""
        dof_error = torch.abs(self.dof_pos - self._get_condition_target_dof_pos())
        standing = torch.linalg.vector_norm(self.commands[:, :2], dim=1) < self.speed_min
        return torch.sum(dof_error, dim=1) * standing

    def _reward_action_jerk(self) -> torch.Tensor:
        second_difference = self.actions - 2.0 * self.last_actions + self.last_last_actions
        return torch.sum(torch.square(second_difference), dim=1) * (self.action_history_steps >= 2).to(self.actions.dtype)

    def _reward_stand_contact(self) -> torch.Tensor:
        threshold = float(getattr(self.cfg.rewards, "stand_contact_threshold", 1.0))
        command_threshold = float(getattr(self.cfg.rewards, "stand_command_threshold", self.speed_min))
        contact = self.contact_forces[:, self.feet_indices, 2] > threshold
        standing = torch.linalg.vector_norm(self.commands[:, :3], dim=1) < command_threshold
        return torch.sum((~contact).to(self.dof_pos.dtype), dim=1) * standing * self._get_gait_reward_active_mask()

    def _reward_base_height(self) -> torch.Tensor:
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self._get_base_height_target())
