"""EL4090 envelope task using the occupied-body API for HAA ranges."""

from __future__ import annotations

import torch
from isaacgym import gymtorch
from typing import Optional

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.el_4090.spider_envelop.el_4090 import EL_4090_ENVELOP
from legged_gym.envs.el_4090.spider_envelop_3.el4090_spider_config import El4090Envelop3Cfg
from legged_gym.envs.el_4090.spider_envelop_3.api_haa_range_network import ApiHaaRangeNetwork
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
        self.haa_range_estimator = self._make_haa_range_estimator(haa_leg_names)
        self.haa_swing_ranges = torch.zeros(
            self.num_envs, len(haa_leg_names), 2, dtype=torch.float,
            device=self.device, requires_grad=False,
        )
        self._haa_range_condition_cache = torch.full(
            (self.num_envs, self.condition_dim), float("nan"), dtype=torch.float,
            device=self.device, requires_grad=False,
        )
        self._refresh_haa_swing_ranges()

    def _make_haa_range_estimator(self, haa_leg_names):
        checkpoint = str(self.cfg.haa_range_network.network_checkpoint).format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR,
        )
        network = ApiHaaRangeNetwork.from_checkpoint(
            checkpoint,
            device=self.device,
            condition_names=self.condition_names,
            condition_low=self.condition_low,
            condition_high=self.condition_high,
        )
        missing_legs = set(haa_leg_names).difference(network.leg_names)
        if missing_legs:
            raise ValueError(f"HAA network is missing legs: {sorted(missing_legs)}")
        self._haa_range_output_indices = torch.tensor(
            [network.leg_names.index(name) for name in haa_leg_names],
            dtype=torch.long,
            device=self.device,
        )
        return network

    @torch.no_grad()
    def _refresh_haa_swing_ranges(self, env_ids=None) -> None:
        if env_ids is None:
            condition = self._get_structure_condition()
            changed = torch.any(condition != self._haa_range_condition_cache, dim=1)
            env_ids = changed.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        condition = self._get_structure_condition()[env_ids]
        ranges = self.haa_range_estimator(condition)[:, self._haa_range_output_indices]
        if ranges.shape != self.haa_swing_ranges[env_ids].shape:
            raise RuntimeError(
                f"HAA estimator returned {tuple(ranges.shape)}, expected "
                f"{tuple(self.haa_swing_ranges[env_ids].shape)}"
            )
        self.haa_swing_ranges[env_ids] = ranges
        self._haa_range_condition_cache[env_ids] = condition

    def _get_condition_target_dof_pos(self) -> torch.Tensor:
        """The sole embedded/controller/observation reference is spider default."""
        return self.default_dof_pos.repeat(self.num_envs, 1)

    def _get_base_height_target(self) -> torch.Tensor:
        return torch.full(
            (self.num_envs,), float(self.cfg.rewards.base_height_spider_target), device=self.device,
        )

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
        """Noise vector for 66-D proprioception plus 12 HAA range values."""
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
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.haa_swing_ranges.reshape(self.num_envs, -1),
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

    @torch.no_grad()
    def _sample_valid_envelope_conditions(self, env_ids: torch.Tensor) -> None:
        """Sample five network inputs and refresh their six predicted HAA ranges."""
        self.envelope_state.sample(env_ids)
        self._refresh_haa_swing_ranges(env_ids)

    @torch.no_grad()
    def set_envelope_condition(self, values, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        target_ids = (torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                      if env_ids is None else env_ids.to(device=self.device, dtype=torch.long))
        updated = self.envelope_state.set(values, env_ids)
        self._refresh_haa_swing_ranges(target_ids)
        return updated

    def _post_physics_step_callback(self) -> None:
        super()._post_physics_step_callback()
        self._refresh_haa_swing_ranges()

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.control.control_type != "P":
            return super()._compute_torques(actions)
        target = actions * self.cfg.control.action_scale + self.default_dof_pos
        torques = self.p_gains * (target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.last_last_actions[env_ids] = 0.0
        self.action_history_steps[env_ids] = 0
        target = self.default_dof_pos.expand(env_ids.numel(), -1)
        self.dof_pos[env_ids] = target
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32),
        )

    def _reward_haa_range_violation(self) -> torch.Tensor:
        """Penalize only positions outside the network-predicted HAA interval."""
        margin = float(getattr(self.cfg.rewards, "haa_range_margin", 0.0))
        lower, upper = self.haa_swing_ranges[..., 0] - margin, self.haa_swing_ranges[..., 1] + margin
        violation = torch.relu(lower - self.dof_pos[:, self.haa_indices]) + torch.relu(self.dof_pos[:, self.haa_indices] - upper)
        normalized_violation = violation / (0.5 * (upper - lower)).clamp_min(0.10)
        return torch.mean(torch.square(torch.clamp(normalized_violation, max=1.0)), dim=1)

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
        """Do not pull HAA back to default while the range task is active."""
        dof_error = torch.abs(self.dof_pos - self.default_dof_pos)
        dof_error[:, self.haa_indices] = 0.0
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
        low = float(getattr(self.cfg.rewards, "base_height_min", 0.47))
        high = float(getattr(self.cfg.rewards, "base_height_max", 0.60))
        if low > high:
            raise ValueError("rewards.base_height_min must not exceed base_height_max")
        deviation = torch.relu(low - base_height) + torch.relu(base_height - high)
        return torch.square(deviation)

    def _draw_envelope_condition_vis(self) -> None:
        """Draw the five-input polygon without invoking the replaced API estimator."""
        if getattr(self, "viewer", None) is None:
            return
        from legged_gym.utils.math_utils import quat_apply_yaw

        env_ids = getattr(self.cfg.commands, "envelope_debug_env_ids", [0])
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        index = {name: column for column, name in enumerate(self.condition_names)}
        color = tuple(getattr(self.cfg.commands, "envelope_debug_color", (0.0, 0.85, 1.0)))
        for env_id in env_ids:
            if env_id < 0 or env_id >= self.num_envs:
                continue
            condition = self._get_structure_condition()[env_id]
            front = condition[index["front_width"]]
            middle = condition[index["middle_width"]]
            back = condition[index["back_width"]]
            forward = condition[index["forward_limit"]]
            backward = condition[index["backward_limit"]]
            local_points = torch.stack((
                torch.stack((forward, front, torch.zeros_like(front))),
                torch.stack((torch.zeros_like(middle), middle, torch.zeros_like(middle))),
                torch.stack((backward, back, torch.zeros_like(back))),
                torch.stack((backward, -back, torch.zeros_like(back))),
                torch.stack((torch.zeros_like(middle), -middle, torch.zeros_like(middle))),
                torch.stack((forward, -front, torch.zeros_like(front))),
            ))
            translation = self.root_states[env_id, :3].clone()
            translation[2] = self.env_origins[env_id, 2] + float(
                getattr(self.cfg.commands, "envelope_debug_ground_z_offset", 0.02)
            )
            points = quat_apply_yaw(
                self.base_quat[env_id].repeat(local_points.shape[0], 1), local_points,
            ) + translation.unsqueeze(0)
            points = points.detach().cpu().numpy()
            vertices, colors = self._make_bold_envelope_lines(
                points, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
                [color] * 6,
                float(getattr(self.cfg.commands, "envelope_debug_line_radius", 0.0)),
                max(1, int(getattr(self.cfg.commands, "envelope_debug_line_samples", 1))),
            )
            self.gym.add_lines(self.viewer, self.envs[env_id], len(vertices), vertices, colors)
