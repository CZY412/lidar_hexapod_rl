"""EL4090 task with five-parameter network-conditioned HAA ranges."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate
from typing import Optional

from legged_gym.envs.el_4090.spider_envelop.el_4090 import EL_4090_ENVELOP
from legged_gym.envs.el_4090.spider_envelop_3.el4090_spider_config import El4090Envelop3Cfg
from legged_gym.envs.el_4090.spider_envelop_3.envelope_range_state import EnvelopeRangeState
from legged_gym.utils.math_utils import quat_apply_yaw


class EL_4090_ENVELOP_3(EL_4090_ENVELOP):
    """Self-contained envelop_3 behavior with directly sampled HAA ranges."""

    cfg: El4090Envelop3Cfg

    def __init__(self, cfg: El4090Envelop3Cfg, sim_params, physics_engine, sim_device, headless, task_name="el4090_envelop_3"):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, task_name)

    def _init_buffers(self) -> None:
        EL_4090_ENVELOP._init_buffers(self)
        # The parent task allocates this legacy morphology buffer. Envelop_3
        # replaces it with an HAA interval-centre bias, without changing the
        # unique static ``default_dof_pos`` buffer.
        del self.mammal_default_dof_pos
        haa_leg_names = tuple(
            self.dof_names[index].split("_", 1)[0] for index in self.haa_indices.tolist()
        )
        checkpoint = Path(getattr(self.cfg.haa_range, "network_path"))
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[1] / checkpoint
        self.haa_range_state = EnvelopeRangeState(
            self.cfg.haa_range, self.num_envs, self.device, checkpoint=str(checkpoint),
        )
        if haa_leg_names != self.haa_range_state.leg_names:
            raise ValueError(f"Configured HAA leg order {self.haa_range_state.leg_names} != Isaac order {haa_leg_names}")
        self.haa_swing_ranges = self.haa_range_state.get()
        self.envelope_parameters = self.haa_range_state.get_conditions()
        self.haa_range_bias = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        # This timer is only reset by runtime envelope replacements.  Initial
        # sampling and episode resets start with the constraint fully active.
        envelope_ramp_time = max(float(getattr(
            self.cfg.rewards, "envelope_constraint_reward_ramp_time", 0.0,
        )), 0.0)
        self.envelope_constraint_transition_time = torch.full(
            (self.num_envs,), envelope_ramp_time, dtype=torch.float,
            device=self.device, requires_grad=False,
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

    def _get_envelope_constraint_transition_mask(self) -> torch.Tensor:
        """Linearly restore the footprint cost after an external replacement."""
        ramp_time = float(getattr(
            self.cfg.rewards, "envelope_constraint_reward_ramp_time", 0.0,
        ))
        if ramp_time <= 0.0:
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        return torch.clamp(self.envelope_constraint_transition_time / ramp_time, 0.0, 1.0)

    def _start_envelope_constraint_transition(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() != 0:
            self.envelope_constraint_transition_time[env_ids] = 0.0

    def _post_physics_step_callback(self) -> None:
        super()._post_physics_step_callback()
        self.envelope_constraint_transition_time += self.dt

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
        enable_stand_commands = bool(getattr(self.cfg.commands, "enable_stand_commands", False))
        stand_probability = float(getattr(self.cfg.commands, "stand_command_probability", 0.05))
        stand_probability = stand_probability if enable_stand_commands else 0.0
        stand_mask = torch.rand(env_ids.numel(), device=self.device) < stand_probability
        stand_ids = env_ids[stand_mask]
        if stand_ids.numel():
            exact_fraction = float(getattr(self.cfg.commands, "stand_command_exact_zero_fraction", 0.5))
            exact_mask = torch.rand(stand_ids.numel(), device=self.device) < exact_fraction
            near_limit = float(getattr(self.cfg.commands, "stand_command_near_zero_limit", 0.05))
            near_commands = (2.0 * torch.rand(stand_ids.numel(), 3, device=self.device) - 1.0) * near_limit
            near_commands[exact_mask] = 0.0
            self.commands[stand_ids, :3] = near_commands
        # Ranges are fixed for an episode and only resampled at reset.
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
        self._start_envelope_constraint_transition(target_ids)
        return ranges

    @torch.no_grad()
    def sample_haa_ranges(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Resample five envelope inputs and infer their HAA intervals."""
        target_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None else env_ids.to(device=self.device, dtype=torch.long)
        )
        ranges = self.haa_range_state.sample(target_ids)
        self._refresh_haa_range_bias(target_ids)
        self.embedded_state_default_dof_pos[target_ids] = self._get_condition_target_dof_pos()[target_ids]
        self._start_envelope_constraint_transition(target_ids)
        return ranges

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.control.control_type != "P":
            return super()._compute_torques(actions)
        target = actions * self.cfg.control.action_scale + self._get_condition_target_dof_pos()
        # HAA actions express gait inside the selected interval, not a way to
        # cancel its centre bias.  Hard-project their *joint targets* so this
        # remains true even when an early policy explores saturated actions.
        target[:, self.haa_indices] = torch.maximum(
            torch.minimum(target[:, self.haa_indices], self.haa_swing_ranges[..., 1]),
            self.haa_swing_ranges[..., 0],
        )
        torques = self.p_gains * (target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.haa_range_state.sample(env_ids)
        self._refresh_haa_range_bias(env_ids)
        self.embedded_state_default_dof_pos[env_ids] = self._get_condition_target_dof_pos()[env_ids]
        # Reset places the robot at the new target, so it is not a runtime
        # reference jump and does not need a grace interval.
        self.envelope_constraint_transition_time[env_ids] = max(float(getattr(
            self.cfg.rewards, "envelope_constraint_reward_ramp_time", 0.0,
        )), 0.0)
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
        # The sampled minimum width is 0.2 rad, so this remains a non-empty
        # interior with the default 0.04 rad margin.
        interior_lower, interior_upper = lower + margin, upper - margin
        signed_interior_distance = torch.minimum(
            self.dof_pos[:, self.haa_indices] - interior_lower,
            interior_upper - self.dof_pos[:, self.haa_indices],
        )
        softness = max(float(getattr(self.cfg.rewards, "haa_range_compliance_softness", 0.04)), 1e-6)
        return torch.mean(torch.sigmoid(signed_interior_distance / softness), dim=1)

    def _reward_envelope_constraint(self) -> torch.Tensor:
        """Return planar foot penetration outside the sampled envelope.

        V3 does not expose its five envelope parameters to the policy, but the
        simulator keeps them in ``envelope_parameters``.  The reward therefore
        uses the same sampled input that was passed to the HAA-range network,
        rather than attempting to reconstruct an envelope from its six output
        intervals.  A positive ``envelope_constraint_margin`` is a tolerance:
        it expands the allowable footprint by that many metres on every face.
        """
        values = self.envelope_parameters
        names = self.haa_range_state.condition_names
        index = {name: position for position, name in enumerate(names)}
        front_width = values[:, index["front_width"]].unsqueeze(1)
        middle_width = values[:, index["middle_width"]].unsqueeze(1)
        back_width = values[:, index["back_width"]].unsqueeze(1)
        x_front = values[:, index["forward_limit"]].unsqueeze(1)
        x_back = values[:, index["backward_limit"]].unsqueeze(1)

        margin = float(getattr(self.cfg.rewards, "envelope_constraint_margin", 0.0))
        x_front = x_front + margin
        x_back = x_back - margin
        front_width = front_width + margin
        middle_width = middle_width + margin
        back_width = back_width + margin

        # Transform each foot from world coordinates into the base-yaw frame.
        foot_rel_world = self.foot_positions - self.base_pos.unsqueeze(1)
        foot_rel_flat = foot_rel_world.reshape(-1, 3)
        base_quat_flat = self.base_quat.unsqueeze(1).expand(-1, foot_rel_world.shape[1], -1).reshape(-1, 4)
        foot_local = quat_apply_yaw(
            quat_conjugate(base_quat_flat), foot_rel_flat,
        ).reshape(self.num_envs, -1, 3)
        foot_x, foot_y = foot_local[..., 0], foot_local[..., 1]

        # The footprint is the symmetric hexagon
        # (x_front, +/-front_width), (0, +/-middle_width),
        # (x_back, +/-back_width).  Clamp x for the lateral interpolation so
        # a foot beyond an end cap incurs independent longitudinal and lateral
        # excess instead of extrapolating a sloped side indefinitely.
        x_on_footprint = torch.clamp(foot_x, min=x_back, max=x_front)
        back_fraction = (x_on_footprint - x_back) / (-x_back).clamp_min(1e-6)
        front_fraction = x_on_footprint / x_front.clamp_min(1e-6)
        half_width_back = back_width + back_fraction * (middle_width - back_width)
        half_width_front = middle_width + front_fraction * (front_width - middle_width)
        half_width = torch.where(x_on_footprint >= 0.0, half_width_front, half_width_back)

        x_excess = torch.relu(x_back - foot_x) + torch.relu(foot_x - x_front)
        y_excess = torch.relu(torch.abs(foot_y) - half_width)
        # Keep the error in metres rather than squaring it. The squared form
        # made ordinary centimetre-scale violations almost invisible after
        # averaging over six feet, so it did not give the policy a useful
        # incentive to recover into the footprint.
        penalty = torch.mean(x_excess + y_excess, dim=1)

        min_command = float(getattr(self.cfg.rewards, "envelope_constraint_min_command", 0.0))
        if min_command > 0.0:
            # Include yaw: turning needs the same footprint protection even
            # when its linear-velocity command is zero.
            moving = torch.linalg.vector_norm(self.commands[:, :3], dim=1) > min_command
            penalty = penalty * moving.to(penalty.dtype)
        return penalty * self._get_envelope_constraint_transition_mask()

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
        # This is deliberately gated by fore/aft command only.  Lateral
        # tracking and yawing already have their own objectives; rewarding
        # HAA activity for them made unnecessary inward/outward leg motions a
        # viable way to collect this reward.
        min_forward_command = float(getattr(
            self.cfg.rewards, "haa_range_swing_min_forward_command", 0.10,
        ))
        forward_commanded = (
            torch.abs(self.commands[:, 0]) > min_forward_command
        ).to(self.dof_pos.dtype)
        return (
            torch.mean(velocity_factor * excursion_factor * interior_factor, dim=1)
            * forward_commanded
        )

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
        """Penalize pose error, joint motion, and action changes at standstill."""
        pose_cost = torch.sum(torch.abs(self.dof_pos - self._get_condition_target_dof_pos()), dim=1)
        velocity_weight = float(getattr(self.cfg.rewards, "stand_still_dof_vel_weight", 0.05))
        action_weight = float(getattr(self.cfg.rewards, "stand_still_action_rate_weight", 0.10))
        velocity_cost = velocity_weight * torch.sum(torch.square(self.dof_vel), dim=1)
        action_cost = action_weight * torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        threshold = float(getattr(self.cfg.rewards, "stand_command_threshold", self.speed_min))
        standing = torch.linalg.vector_norm(self.commands[:, :3], dim=1) < threshold
        return (pose_cost + velocity_cost + action_cost) * standing

    def _reward_stand_contact(self) -> torch.Tensor:
        """At rest, require both legs in every front/middle/rear pair to load."""
        threshold = float(getattr(self.cfg.rewards, "stand_contact_threshold", 10.0))
        pair_threshold = float(getattr(self.cfg.rewards, "stand_pair_force_threshold", 30.0))
        command_threshold = float(getattr(self.cfg.rewards, "stand_command_threshold", self.speed_min))
        normal_forces = self.contact_forces[:, self.feet_indices, 2].clamp_min(0.0)
        loaded_feet = normal_forces >= threshold
        standing = torch.linalg.vector_norm(self.commands[:, :3], dim=1) < command_threshold
        # Foot order is [LB, LF, LM, RB, RF, RM].  Each pair must have both
        # feet loaded and enough combined force; a middle-leg biped therefore
        # cannot pass as a stationary six-leg pose.
        leg_pairs = ((1, 4), (2, 5), (0, 3))  # front, middle, rear
        missing_loaded_feet = (~loaded_feet).sum(dim=1).to(normal_forces.dtype)
        invalid_pairs = torch.zeros(self.num_envs, device=self.device, dtype=normal_forces.dtype)
        for pair in leg_pairs:
            pair_loaded = loaded_feet[:, pair].all(dim=1)
            pair_loaded &= normal_forces[:, pair].sum(dim=1) >= pair_threshold
            invalid_pairs += (~pair_loaded).to(normal_forces.dtype)
        return (missing_loaded_feet + invalid_pairs) * standing * self._get_gait_reward_active_mask()

    def _reward_tripod_contact_pattern(self) -> torch.Tensor:
        """Require the active support phase to be a complete front/middle/rear tripod."""
        contact_threshold = float(getattr(self.cfg.rewards, "tripod_contact_threshold", 1.0))
        support_threshold = float(getattr(self.cfg.rewards, "support_foot_force_threshold", 10.0))
        min_command = float(getattr(self.cfg.rewards, "tripod_contact_min_command", 0.1))
        normal_forces = self.contact_forces[:, self.feet_indices, 2].clamp_min(0.0)
        contact = (normal_forces > contact_threshold).to(self.dof_pos.dtype)
        supporting = (normal_forces >= support_threshold).to(self.dof_pos.dtype)
        group1, group2 = contact[:, self.tripod_group1_indices], contact[:, self.tripod_group2_indices]
        group1_same = torch.var(group1, dim=1, unbiased=False)
        group2_same = torch.var(group2, dim=1, unbiased=False)
        group_opposite = torch.square(group1.mean(dim=1) + group2.mean(dim=1) - 1.0)

        forces_g1 = normal_forces[:, self.tripod_group1_indices]
        forces_g2 = normal_forces[:, self.tripod_group2_indices]
        active_support = torch.where(
            (forces_g1.sum(dim=1) >= forces_g2.sum(dim=1)).unsqueeze(1),
            supporting[:, self.tripod_group1_indices],
            supporting[:, self.tripod_group2_indices],
        )
        missing_support_legs = 3.0 - active_support.sum(dim=1)
        moving = torch.linalg.vector_norm(self.commands[:, :2], dim=1) > min_command
        return (
            group1_same + group2_same + group_opposite + missing_support_legs
        ) * moving.to(self.dof_pos.dtype) * self._get_gait_reward_active_mask()

    def _reward_contact_force_balance(self) -> torch.Tensor:
        """Penalize missing or uneven load within the active support tripod."""
        min_command = float(getattr(self.cfg.rewards, "tripod_contact_min_command", 0.1))
        min_force = float(getattr(self.cfg.rewards, "support_foot_force_threshold", 10.0))
        normal_forces = self.contact_forces[:, self.feet_indices, 2].clamp_min(0.0)
        forces_g1 = normal_forces[:, self.tripod_group1_indices]
        forces_g2 = normal_forces[:, self.tripod_group2_indices]
        active_forces = torch.where(
            (forces_g1.sum(dim=1) >= forces_g2.sum(dim=1)).unsqueeze(1), forces_g1, forces_g2
        )
        minimum_load_error = torch.relu(min_force - active_forces).sum(dim=1) / min_force
        force_shares = active_forces / active_forces.sum(dim=1, keepdim=True).clamp_min(1e-6)
        equal_share_error = 3.0 * torch.square(force_shares - 1.0 / 3.0).sum(dim=1)
        moving = torch.linalg.vector_norm(self.commands[:, :2], dim=1) > min_command
        return (
            minimum_load_error + equal_share_error
        ) * moving.to(self.dof_pos.dtype) * self._get_gait_reward_active_mask()

    def _envelope_debug_points_world(self, env_id: int) -> np.ndarray:
        """Convert v3's five sampled envelope values into a ground footprint."""
        condition = self.envelope_parameters[env_id]
        names = self.haa_range_state.condition_names
        value = {name: condition[index] for index, name in enumerate(names)}
        local_z = torch.zeros((), dtype=torch.float, device=self.device)
        local_points = torch.stack((
            torch.stack((value["forward_limit"], value["front_width"], local_z)),
            torch.stack((torch.zeros_like(local_z), value["middle_width"], local_z)),
            torch.stack((value["backward_limit"], value["back_width"], local_z)),
            torch.stack((value["backward_limit"], -value["back_width"], local_z)),
            torch.stack((torch.zeros_like(local_z), -value["middle_width"], local_z)),
            torch.stack((value["forward_limit"], -value["front_width"], local_z)),
        ))
        from legged_gym.utils.math_utils import quat_apply_yaw

        translation = self.root_states[env_id, :3].clone()
        translation[2] = self.env_origins[env_id, 2] + float(
            getattr(self.cfg.commands, "envelope_debug_ground_z_offset", 0.03)
        )
        return (quat_apply_yaw(
            self.base_quat[env_id].repeat(local_points.shape[0], 1), local_points,
        ) + translation.unsqueeze(0)).detach().cpu().numpy()

    def _draw_five_parameter_envelope_vis(self) -> None:
        """Draw a thick cyan footprint for the current v3 envelope parameters."""
        if getattr(self, "viewer", None) is None:
            return
        env_ids = getattr(self.cfg.commands, "envelope_debug_env_ids", [0])
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        color = tuple(getattr(self.cfg.commands, "envelope_debug_color", (0.0, 0.95, 1.0)))
        radius = float(getattr(self.cfg.commands, "envelope_debug_line_radius", 0.025))
        samples = max(1, int(getattr(self.cfg.commands, "envelope_debug_line_samples", 12)))
        edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4))
        for env_id in env_ids:
            if env_id < 0 or env_id >= self.num_envs:
                continue
            vertices, colors = self._make_bold_envelope_lines(
                self._envelope_debug_points_world(env_id), edges, [color] * len(edges), radius, samples,
            )
            self.gym.add_lines(self.viewer, self.envs[env_id], len(colors), vertices, colors)

    def _draw_envelope_condition_vis(self) -> None:
        """Draw the v3 footprint plus the selected leg's sampled HAA interval.

        Cyan: five-parameter envelope footprint. Dark green: allowed HAA arc
        and endpoint spokes. Red: instantaneous HAA position.
        """
        self._draw_five_parameter_envelope_vis()
        if getattr(self, "viewer", None) is None:
            return
        env_ids = getattr(self.cfg.commands, "haa_range_debug_env_ids", [0])
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        leg_name = str(getattr(self.cfg.commands, "haa_range_debug_leg", "LF"))
        radius = float(getattr(self.cfg.commands, "haa_range_debug_radius", 0.22))
        samples = max(3, int(getattr(self.cfg.commands, "haa_range_debug_samples", 33)))
        arc_color = tuple(getattr(self.cfg.commands, "haa_range_debug_arc_color", (0.0, 0.42, 0.0)))
        endpoint_color = tuple(getattr(self.cfg.commands, "haa_range_debug_endpoint_color", (0.0, 0.24, 0.0)))
        marker_color = tuple(getattr(self.cfg.commands, "haa_range_debug_marker_color", (0.90, 0.08, 0.0)))
        line_radius = float(getattr(self.cfg.commands, "haa_range_debug_line_radius", 0.014))
        line_samples = max(1, int(getattr(self.cfg.commands, "haa_range_debug_line_samples", 8)))
        self._ensure_haa_range_visualizer()
        from legged_gym.utils.math_utils import quat_apply_yaw

        leg_index = self._haa_vis_leg_names.index(leg_name)
        dof_index = {name: index for index, name in enumerate(self.dof_names)}
        for env_id in env_ids:
            if env_id < 0 or env_id >= self.num_envs:
                continue
            q = torch.tensor(
                [self.dof_pos[env_id, dof_index[name]].item() for name in self._haa_vis_joint_names],
                dtype=torch.float32,
            )
            ranges = torch.stack([
                self.haa_swing_ranges[env_id, self.haa_range_state.leg_names.index(leg)].detach().cpu()
                for leg in self._haa_vis_leg_names
            ])
            origins, arcs, markers = self._haa_arc_geometry(
                self._haa_vis_kinematics, q, ranges, radius=radius, samples=samples,
            )
            local_points = torch.cat((
                origins[leg_index:leg_index + 1], arcs[leg_index], markers[leg_index:leg_index + 1],
            ))
            world_points = quat_apply_yaw(
                self.base_quat[env_id].detach().cpu().repeat(local_points.shape[0], 1), local_points,
            ) + self.root_states[env_id, :3].detach().cpu().unsqueeze(0)
            origin = world_points[0].numpy()
            arc = world_points[1:-1].numpy()
            marker = world_points[-1].numpy()
            # Reuse envelop_2's visible multi-line tube treatment instead of
            # Isaac Gym's otherwise hard-to-see one-pixel debug lines.
            points = np.concatenate((arc, origin[None], marker[None]), axis=0)
            origin_index, marker_index = len(arc), len(arc) + 1
            edges = [(index, index + 1) for index in range(len(arc) - 1)]
            edge_colors = [arc_color] * len(edges)
            edges.extend(((origin_index, 0), (origin_index, len(arc) - 1), (origin_index, marker_index)))
            edge_colors.extend((endpoint_color, endpoint_color, marker_color))
            vertices, colors = self._make_bold_envelope_lines(
                points, edges, edge_colors, line_radius, line_samples,
            )
            self.gym.add_lines(
                self.viewer,
                self.envs[env_id],
                len(colors),
                vertices,
                colors,
            )

    def _ensure_haa_range_visualizer(self) -> None:
        if hasattr(self, "_haa_vis_kinematics"):
            return
        source_root = Path(__file__).resolve().parents[3] / "utils" / "envelop" / "el4090_envelope" / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from el4090_envelope import BatchedUrdfKinematics, load_urdf_joints
        from el4090_envelope.geometry import haa_arc_geometry
        from el4090_envelope.model import EL4090_JOINT_NAMES, EL4090_LEG_NAMES

        urdf = Path(__file__).resolve().parents[4] / "resources" / "robots" / "el_4090" / "urdf" / "el_4090.urdf"
        self._haa_vis_kinematics = BatchedUrdfKinematics(load_urdf_joints(urdf))
        self._haa_vis_joint_names = tuple(EL4090_JOINT_NAMES)
        self._haa_vis_leg_names = tuple(EL4090_LEG_NAMES)
        self._haa_arc_geometry = haa_arc_geometry

    def _reward_base_height(self) -> torch.Tensor:
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self._get_base_height_target())

    def _reward_base_height_minimum(self) -> torch.Tensor:
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        minimum = float(getattr(self.cfg.rewards, "minimum_base_height", 0.50))
        return torch.square(torch.clamp(minimum - base_height, min=0.0))
