"""M1 envelope_adaptive_2 environment (``EL_4090_EA2``).

This module integrates T1-T6 (map, path, hex geometry, range image, Airy
mount, LiDAR noise) into a simplified legged_gym ``BaseTask`` with no robot
actor.  The policy sees a 190-dim observation (187 fixed range channels + 3
ego-motion) and emits 5 raw envelope parameters which are mapped in the
environment to the spider_envelop ranges.

Pure helpers are kept at module level so ``tests/ea2/test_ea2_env_helpers.py``
can exercise height/wobble, heading, rewards, observation assembly and the
empty-frame reset without creating an Isaac Gym simulation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
    El4090EA2Cfg,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    _hex_sample_violations,
    compute_hex_vertices,
    hex_collision_terms,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
    compute_direct_oracle_params_with_stats,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.map_generator import (
    generate_map,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.path_batch import (
    DEFAULT_MAX_CORNERS,
    DEFAULT_MAX_POINTS,
    PathBatch,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.path_planner import (
    PathCfg,
    PathData,
    plan_path,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.range_image import (
    build_selected_range_image,
    extract_selected_range_image,
    range_image_observation,
)
from legged_gym.utils.envelop.network.haa_swing_range import (
    load_envelope_condition_spec,
)

# ---------------------------------------------------------------------------
# Module-level pure helpers (Isaac-free, unit-testable)
# ---------------------------------------------------------------------------


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle/tensor to ``[-pi, pi)`` using PyTorch semantics."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_quat_from_heading(heading: torch.Tensor) -> torch.Tensor:
    """Build a z-up yaw quaternion ``[x, y, z, w]`` from a body heading.

    M1 has no pitch/roll, so the robot/body orientation is fully described by
    ``heading``.  This helper is the single source of truth for converting
    heading to the quaternion used by LiDAR raycasting and debug drawing.
    """
    half = heading * 0.5
    return torch.stack(
        [
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
            torch.cos(half),
        ],
        dim=-1,
    )


def height_step(
    height_filter: torch.Tensor,
    height_target: torch.Tensor,
    wobble_state: torch.Tensor,
    noise: torch.Tensor,
    min_m: float,
    max_m: float,
    dt: float,
    tau_s: float,
    wobble_fc_hz: float,
    wobble_amp: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the time-varying carrier height model one control step.

    Implements README 2.2.7: a first-order low-pass filter toward a
    periodically resampled target plus a separate 1 Hz low-pass AR wobble.

    Returns ``(height_filter, wobble_state, height)``.  ``wobble_state`` is a
    normalized AR(1) state; the returned height is clipped to ``[min_m,
    max_m]``.
    """
    alpha = 1.0 - math.exp(-dt / max(tau_s, 1e-6))
    height_filter = height_filter + alpha * (height_target - height_filter)
    a = math.exp(-2.0 * math.pi * max(wobble_fc_hz, 0.0) * dt)
    wobble_state = a * wobble_state + math.sqrt(max(1.0 - a * a, 0.0)) * noise
    height = (height_filter + wobble_amp * wobble_state).clamp(min_m, max_m)
    return height_filter, wobble_state, height


def sway_update(
    state: torch.Tensor,
    amp: float,
    dt: float,
    fc_hz: float,
    noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Advance a first-order low-pass sway state.

    The returned offset is ``amp * state`` where ``state`` is an AR(1) process
    driven by unit-variance noise.  Used for both lateral position sway and
    heading sway (README 2.2.7).
    """
    a = math.exp(-2.0 * math.pi * max(fc_hz, 0.0) * dt)
    state = a * state + math.sqrt(max(1.0 - a * a, 0.0)) * noise
    return state, amp * state


def heading_update(
    heading: torch.Tensor,
    tangent: torch.Tensor,
    tangent_rate: torch.Tensor,
    delta_target: torch.Tensor,
    dt: float,
    k_p: float,
    omega_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tangent-relative heading controller (README 2.2.5).

    Returns ``(heading, omega, delta_actual)``.
    """
    delta_actual = wrap_to_pi(heading - tangent)
    omega_cmd = tangent_rate + k_p * wrap_to_pi(delta_target - delta_actual)
    omega = torch.clamp(
        torch.as_tensor(omega_cmd, dtype=torch.float32), -omega_max, omega_max
    )
    heading = wrap_to_pi(heading + omega * dt)
    delta_actual = wrap_to_pi(heading - tangent)
    return heading, omega, delta_actual


def ego_motion(
    v: torch.Tensor,
    heading: torch.Tensor,
    tangent: torch.Tensor,
    omega: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Body-frame ego motion from path tangent tracking (README 2.2.5/2.4)."""
    delta = heading - tangent
    return v * torch.cos(delta), v * torch.sin(delta), omega


def default_params(
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Return the envelope midpoint (action=0 corresponds to this pose)."""
    return 0.5 * (low + high)


def envelope_action_scale(
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float,
    action_max: float,
) -> torch.Tensor:
    """Per-parameter linear action scale, mirroring standard legged_gym.

    ``action_max`` is the raw-action radius that reaches the soft bounds;
    ``soft_dof_pos_limit`` reserves a small band before the true hard clamp.
    """
    span = high - low
    return span * soft_dof_pos_limit / (2.0 * action_max)


def linear_params_target(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float,
    action_max: float,
) -> torch.Tensor:
    """Compute the unclamped linear envelope target from raw actions."""
    default = default_params(low, high)
    scale = envelope_action_scale(low, high, soft_dof_pos_limit, action_max)
    return default + actions * scale


def map_actions_to_params(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float = 0.9,
    action_max: float = 4.0,
) -> torch.Tensor:
    """Map raw actions to hard-clamped envelope parameters (linear + clamp).

    Thin wrapper around :func:`map_actions_to_params_with_target`; the
    unclamped target is available through that variant for the soft-limit
    reward.
    """
    params, _ = map_actions_to_params_with_target(
        actions, low, high, soft_dof_pos_limit, action_max
    )
    return params


def map_actions_to_params_with_target(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float = 0.9,
    action_max: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(clamped_params, unclamped_target)`` for linear mapping."""
    target = linear_params_target(
        actions, low, high, soft_dof_pos_limit, action_max
    )
    return torch.clamp(target, low, high), target


def envelope_limit_violation(
    target: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float = 0.9,
) -> torch.Tensor:
    """Soft out-of-limits penalty for the unclamped linear target.

    The soft bounds are the central ``soft_dof_pos_limit`` fraction of the
    ``[low, high]`` range.  Penalizing the unclamped target (rather than the
    clamped params) makes the reward keep providing a gradient when the policy
    tries to push beyond the working range.
    """
    margin = (1.0 - soft_dof_pos_limit) * (high - low) / 2.0
    soft_low = low + margin
    soft_high = high - margin

    low_over = torch.relu(soft_low - target)
    high_over = torch.relu(target - soft_high)
    return torch.sum(low_over ** 2 + high_over ** 2, dim=-1)


def normalized_envelope_params(
    params5: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Return each envelope parameter normalized to ``[0, 1]``.

    ``backward_limit`` is normalized in its physical direction so that a
    more negative value (larger rear extent) maps to 1, matching
    :func:`potential_reward`.
    """
    low5 = low[:5]
    high5 = high[:5]
    norm = (params5 - low5) / (high5 - low5).clamp_min(1e-6)
    # backward_limit: value_low = -high, value_high = -low.
    norm = norm.clone()
    norm[..., 4] = (-params5[..., 4] - (-high5[4])) / (
        (-low5[4]) - (-high5[4])
    ).clamp_min(1e-6)
    return norm.clamp(0.0, 1.0)


def potential_reward(
    params5: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Normalized-parameter mean potential (README 2.8).

    ``backward_limit`` is negated before normalization, matching the physical
    direction used by ``apply_env_morphology_priors``.
    """
    return normalized_envelope_params(params5, low, high).mean(dim=-1)


def raw_action_rate_term(
    actions: torch.Tensor,
    last_actions: torch.Tensor,
) -> torch.Tensor:
    """Sum squared change of raw actions, mirroring standard legged_gym.

    Raw-action rate is used because the mapped parameters saturate at the hard
    clamp; a mapped-parameter rate would become ~0 in the saturated region and
    would not prevent raw-action drift.
    """
    return torch.sum((actions - last_actions) ** 2, dim=-1)


def assemble_observation(
    range_image: torch.Tensor,
    ego: torch.Tensor,
    max_range: float = ea2c.EA2_RANGE_MAX_M,
    ego_scales: Sequence[float] = (1.5, 1.0, 1.5),
) -> torch.Tensor:
    """Concatenate normalized range image and normalized ego-motion."""
    normalized_range = range_image_observation(range_image, max_range)
    scales = normalized_range.new_tensor(ego_scales, dtype=torch.float32)
    normalized_ego = ego / scales
    return torch.cat([normalized_range, normalized_ego], dim=-1)


def empty_range_image(
    num_envs: int,
    max_range: float = ea2c.EA2_RANGE_MAX_M,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return an empty range-image buffer filled with ``max_range``."""
    return torch.full(
        (int(num_envs), ea2c.EA2_RANGE_DIM),
        max_range,
        dtype=torch.float32,
        device=device,
    )


def refresh_range_image_from_scan(
    range_image: torch.Tensor,
    fresh: torch.Tensor,
    stale: torch.Tensor,
) -> torch.Tensor:
    """Copy a global LiDAR scan into the live range-image buffer.

    README 2.4 empty-frame contract: an env reset between two global scans
    keeps the all-``max_range`` empty frame only until the next global scan.
    At that scan it must receive the freshly computed aggregate from its new
    pose.  Therefore this helper copies the whole fresh frame for every env
    (including reset-before-scan envs) and only then clears the stale marker;
    it deliberately does NOT replace ``fresh`` rows with ``max_range``.

    This is an Isaac-free helper so the contract can be unit-tested without
    creating a simulation.  It mutates and returns ``range_image``.
    """
    range_image.copy_(fresh.to(device=range_image.device))
    stale[:] = False
    return range_image


def selected_channel_mask(
    dists: torch.Tensor,
    selected_indices: torch.Tensor,
    is_reduced: bool,
    far_plane: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(red, green)`` boolean masks for debug visualization.

    In reduced mode the debug cloud contains only the 187 selected channels,
    so every valid hit is red.  In full mode the debug cloud contains the
    complete 86,400-ray Airy cloud, so only ``selected_indices`` are red and
    the remaining valid hits are green.

    Args:
        dists: Debug ray distances, shape ``(N,)``.
        selected_indices: Full Airy indices of the 187 selected channels.
        is_reduced: Whether ``dists`` is the reduced 187-channel cloud.
        far_plane: Sensor no-hit distance.

    Returns:
        ``(red, green)`` boolean masks with the same shape as ``dists``.
    """
    hit = dists < (float(far_plane) - 1e-3)
    if is_reduced or dists.shape[0] == ea2c.EA2_RANGE_DIM:
        selected = torch.ones_like(hit)
    else:
        selected = torch.zeros_like(hit)
        selected[selected_indices.to(device=dists.device)] = True
    red = selected & hit
    green = hit & ~red
    return red, green


class EL_4090_EA2(BaseTask):
    __doc__ = "Simplified BaseTask for envelope perception (M1, no robot actor)."

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, task_name='el4090_ea2'):
        self.cfg = cfg
        self.sim_params = sim_params
        self.task_name = task_name
        self.dt = float(cfg.sim.dt)
        self.max_episode_length = int(math.ceil(float(cfg.env.episode_length_s) / self.dt))
        self._spec = load_envelope_condition_spec(ea2c.ENVELOPE_SPEC_CONFIG_PATH)
        self._envelope_low = torch.tensor((self._spec.low[:5]),
          dtype=(torch.float32))
        self._envelope_high = torch.tensor((self._spec.high[:5]),
          dtype=(torch.float32))
        self.reward_scales = {'potential':float(cfg.rewards.scales.potential), 
         'collision':float(cfg.rewards.scales.collision), 
         'action_rate':float(cfg.rewards.scales.action_rate), 
         'envelope_limits':float(cfg.rewards.scales.envelope_limits), 
         'oracle_mse':float(cfg.rewards.scales.oracle_mse)}
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_buffers()
        self._init_lidar()

    def create_sim(self):
        """Create Isaac sim, a ground plane, empty shared-origin envs and
        static visual obstacle actors (only in env 0)."""
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        map_cfg = ea2c.MapGenCfg(size_m=(float(self.cfg.map.size_m)),
          resolution_m=(float(self.cfg.map.resolution_m)),
          grid_shape=(tuple(self.cfg.map.grid_shape)),
          boundary_occupied=(bool(self.cfg.map.boundary_occupied)),
          ground_margin_m=(float(self.cfg.map.ground_margin_m)),
          inflation_m=(float(self.cfg.map.inflation_m)),
          inflation_cells=(int(self.cfg.map.inflation_cells)),
          n_tiles=(int(self.cfg.map.n_tiles)),
          tile_size_m=(float(self.cfg.map.tile_size_m)),
          border_size_m=(float(self.cfg.map.border_size_m)),
          min_free_component_ratio=(float(self.cfg.map.min_free_component_ratio)),
          max_gen_attempts=(int(self.cfg.map.max_gen_attempts)),
          n_validation_paths=(int(self.cfg.map.n_validation_paths)),
          min_solved_ratio=(float(self.cfg.map.min_solved_ratio)),
          path_near_obstacle_ratio=(float(self.cfg.map.path_near_obstacle_ratio)),
          near_obstacle_range=(tuple(self.cfg.map.near_obstacle_range)),
          require_constraint_primitive=(bool(self.cfg.map.require_constraint_primitive)))
        pillar_cfg = ea2c.PillarFieldCfg(count_min=(int(self.cfg.obstacles.pillar_count_min)),
          count_max=(int(self.cfg.obstacles.pillar_count_max)),
          size_x_min=(float(self.cfg.obstacles.pillar_size_x_min)),
          size_x_max=(float(self.cfg.obstacles.pillar_size_x_max)),
          size_y_min=(float(self.cfg.obstacles.pillar_size_y_min)),
          size_y_max=(float(self.cfg.obstacles.pillar_size_y_max)),
          height_min=(float(self.cfg.obstacles.pillar_height_min)),
          height_max=(float(self.cfg.obstacles.pillar_height_max)),
          min_separation=(float(self.cfg.obstacles.pillar_min_separation)),
          center_clear_radius=(float(self.cfg.obstacles.pillar_center_clear_radius)),
          spawn_radius=(float(self.cfg.obstacles.pillar_spawn_radius)),
          allow_height_variation=(bool(self.cfg.obstacles.pillar_allow_height_variation)))
        map_seed = int(getattr(self.cfg, "seed", 42))
        self.map_data = generate_map(map_cfg, pillar_cfg, seed=map_seed)
        self.occupancy = self.map_data.occupancy
        self.inflated = self.map_data.inflated
        self.distance_field = torch.as_tensor((self.map_data.distance_field),
          dtype=(torch.float32),
          device=(self.device))
        from scipy.ndimage import label as _label
        from scipy.ndimage import sum as _nd_sum
        free = self.inflated == 0
        labels, n_components = _label(free)
        if n_components == 0:
            raise RuntimeError("map has no inflated-free cells")
        sizes = _nd_sum(free, labels, index=(range(1, n_components + 1)))
        largest_label = int(np.argmax(sizes)) + 1
        self._free_cells = np.argwhere(labels == largest_label)
        half_tile = float(self.cfg.map.size_m) / 2.0 - float(self.cfg.map.border_size_m)
        spawn_xs = ea2c.EA2_WORLD_MIN_XY + (self._free_cells[:, 1] + 0.5) * ea2c.EA2_RESOLUTION_M
        spawn_ys = ea2c.EA2_WORLD_MIN_XY + (self._free_cells[:, 0] + 0.5) * ea2c.EA2_RESOLUTION_M
        spawn_mask = (spawn_xs >= -half_tile) & (spawn_xs <= half_tile) & (spawn_ys >= -half_tile) & (spawn_ys <= half_tile)
        self._spawn_cells = self._free_cells[spawn_mask]
        if self._spawn_cells.shape[0] == 0:
            raise RuntimeError("no spawn cells inside the 4x4 tile area")
        plane_params = self.gymapi().PlaneParams()
        plane_params.normal = self.gymapi().Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        env_lower = self.gymapi().Vec3(-40.0, -40.0, -20.0)
        env_upper = self.gymapi().Vec3(40.0, 40.0, 20.0)
        num_per_row = max(1, int(math.isqrt(self.num_envs)))
        self.envs = []
        self.env_origins = torch.zeros((self.num_envs),
          3, dtype=(torch.float32), device=(self.device))
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_handle)
            if i == 0:
                self._add_visual_obstacles(env_index=0)

    def gymapi(self):
        """Lazily return the Isaac Gym Python module (kept as a method so the
        pure-helper tests never need Isaac unless the class is instantiated)."""
        import isaacgym
        return isaacgym.gymapi

    def _add_visual_obstacles(self, env_index: "int"=0):
        """Add static box actors to one env for visualization only.

        Warp mesh remains the authoritative raycast geometry; these actors do
        not participate in collision or raycast.
        """
        if not self.envs:
            return
        env_handle = self.envs[env_index]
        gymapi = self.gymapi()
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        for rect in self.map_data.rects:
            sx, sy = float(rect.size[0]), float(rect.size[1])
            sz = float(rect.height)
            asset = self.gym.create_box(self.sim, sx, sy, sz, asset_options)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(float(rect.center[0]), float(rect.center[1]), sz / 2.0)
            pose.r = gymapi.Quat.from_euler_zyx(float(rect.yaw), 0.0, 0.0)
            self.gym.create_actor(env_handle, asset, pose, "ea2_visual_wall", env_index, 0, 0)
        else:
            for pillar in self.map_data.pillars:
                side = 2.0 * float(pillar.radius)
                sz = float(pillar.height)
                asset = self.gym.create_box(self.sim, side, side, sz, asset_options)
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(float(pillar.center[0]), float(pillar.center[1]), sz / 2.0)
                self.gym.create_actor(env_handle, asset, pose, "ea2_visual_pillar", env_index, 0, 0)

    def _init_buffers(self):
        device = self.device
        n = self.num_envs
        self._lidar_decimation = max(1, int(round(1.0 / (self.dt * float(self.cfg.lidar.update_frequency_hz)))))
        self._lidar_timer = self._lidar_decimation - 1
        self.common_step_counter = 0
        self.paths = [
         None] * n
        self._path_batch = PathBatch(num_envs=n,
          max_points=DEFAULT_MAX_POINTS,
          max_corners=DEFAULT_MAX_CORNERS,
          device=device)
        self._envelope_low_dev = self._envelope_low.to(device)
        self._envelope_high_dev = self._envelope_high.to(device)
        self.s = torch.zeros(n, dtype=(torch.float32), device=device)
        self.v = torch.zeros(n, dtype=(torch.float32), device=device)
        self.heading = torch.zeros(n, dtype=(torch.float32), device=device)
        self.tangent = torch.zeros(n, dtype=(torch.float32), device=device)
        self.tangent_rate = torch.zeros(n, dtype=(torch.float32), device=device)
        self.delta_target = torch.zeros(n, dtype=(torch.float32), device=device)
        self.delta_actual = torch.zeros(n, dtype=(torch.float32), device=device)
        self.omega = torch.zeros(n, dtype=(torch.float32), device=device)
        self.base_pos = torch.zeros(n, 3, dtype=(torch.float32), device=device)
        self.height_filter = torch.zeros(n, dtype=(torch.float32), device=device)
        self.height_target = torch.zeros(n, dtype=(torch.float32), device=device)
        self.height_wobble_state = torch.zeros(n,
          dtype=(torch.float32), device=device)
        self.pos_sway_state = torch.zeros(n, dtype=(torch.float32), device=device)
        self.heading_sway_state = torch.zeros(n,
          dtype=(torch.float32), device=device)
        self.pos_sway_amp = torch.zeros(n, dtype=(torch.float32), device=device)
        self.heading_sway_amp = torch.zeros(n,
          dtype=(torch.float32), device=device)
        self.height_wobble_amp = torch.zeros(n,
          dtype=(torch.float32), device=device)
        self.actions = torch.zeros(n,
          (self.num_actions), dtype=(torch.float32), device=device)
        self.actions_mapped = torch.zeros(n,
          (self.num_actions), dtype=(torch.float32), device=device)
        self.actions_target = torch.zeros(n,
          (self.num_actions), dtype=(torch.float32), device=device)
        self.last_actions_raw = torch.zeros(n,
          (self.num_actions), dtype=(torch.float32), device=device)
        self.range_image = empty_range_image(n, ea2c.EA2_RANGE_MAX_M, device)
        self.range_image_stale = torch.ones(n, dtype=(torch.bool), device=device)
        self.sensor_pos_tensor = torch.zeros(n,
          3, dtype=(torch.float32), device=device)
        self.sensor_quat_tensor = torch.zeros(n,
          4, dtype=(torch.float32), device=device)
        self.ego_motion = torch.zeros(n,
          3, dtype=(torch.float32), device=device)
        self._replan_reset_buf = torch.zeros(n,
          dtype=(torch.bool), device=device)
        self._turn_in_place = torch.zeros(n,
          dtype=(torch.bool), device=device)
        self._turn_target = torch.zeros(n,
          dtype=(torch.float32), device=device)
        self._debug_env_ids = [int(i) for i in getattr(self.cfg.lidar, "debug_env_ids", [0])]
        self._debug_point_stride = int(getattr(self.cfg.lidar, "debug_point_stride", 1))
        self._debug_points = None
        self._debug_dists = None
        self._debug_is_reduced = False
        self.episode_sums = {name: torch.zeros(n, dtype=(torch.float32), device=device) for name in self.reward_scales.keys()}
        self.episode_metrics = {'collision_hard_max':torch.zeros(n, dtype=(torch.float32), device=device), 
         'raw_action_abs_mean':torch.zeros(n, dtype=(torch.float32), device=device), 
         'envelope_limits_active_ratio':torch.zeros(n,
           dtype=(torch.float32), device=device), 
         'oracle_unsafe_ratio':torch.zeros(n,
           dtype=(torch.float32), device=device), 
         'oracle_unsafe_before_ratio':torch.zeros(n,
           dtype=(torch.float32), device=device), 
         'oracle_potential':torch.zeros(n,
           dtype=(torch.float32), device=device),
         'oracle_mse_front_width':torch.zeros(n,
           dtype=(torch.float32), device=device),
         'oracle_mse_middle_width':torch.zeros(n,
           dtype=(torch.float32), device=device),
         'oracle_mse_back_width':torch.zeros(n,
           dtype=(torch.float32), device=device),
         'oracle_mse_forward_limit':torch.zeros(n,
           dtype=(torch.float32), device=device),
         'oracle_mse_backward_limit':torch.zeros(n,
           dtype=(torch.float32), device=device),
         'oracle_snap_ratio':torch.zeros(n,
           dtype=(torch.float32), device=device)}
        from legged_gym.envs.el_4090.envelope_adaptive_2.target_smoother import (
            RateLimitedOracle,
        )
        if bool(getattr(self.cfg.envelope, "target_rate_limit", False)):
            def _target_safety_check(candidate):
                viol = _hex_sample_violations(
                    candidate,
                    self.heading,
                    self.base_pos[:, :2],
                    self.distance_field,
                    margin=float(self.cfg.envelope.margin),
                    soft_margin=float(self.cfg.envelope.soft_margin),
                )
                return viol.max(dim=-1).values > 0.05

            self._oracle_smoother = RateLimitedOracle(
                num_envs=n,
                dt=float(self.cfg.sim.dt),
                device=device,
                low=self._envelope_low_dev,
                high=self._envelope_high_dev,
                shrink_rate=float(self.cfg.envelope.target_shrink_rate),
                grow_rate=float(self.cfg.envelope.target_grow_rate),
                cooldown_seconds=float(self.cfg.envelope.target_cooldown_seconds),
                safety_check=_target_safety_check,
            )
        else:
            self._oracle_smoother = None
        self._rng = np.random.default_rng(int(getattr(self.cfg, "seed", 42)) + 1)

    def _init_lidar(self):
        """Initialize the Warp mesh and the shared LidarSensor instance."""
        import warp as wp
        from isaacgym.torch_utils import quat_from_euler_xyz
        from legged_gym.utils.LidarSensor import LidarConfig, LidarSensor, LidarType
        wp.init()
        vertices = torch.from_numpy(self.map_data.vertices).to(self.device)
        triangles = np.asarray((self.map_data.triangles), dtype=(np.int32)).flatten()
        self._wp_mesh = wp.Mesh(points=wp.from_torch(vertices, dtype=(wp.vec3)),
          indices=wp.from_numpy(triangles, dtype=(wp.int32), device=(self.device)))
        self.mesh_ids = wp.array([
         self._wp_mesh.id],
          dtype=(wp.uint64), device=(self.device))
        use_full_lidar = not bool(getattr(self.cfg.lidar, "use_reduced_raycast", True))
        if use_full_lidar:
            lidar_cfg = LidarConfig(sensor_type=(LidarType.AIRY),
              dt=(float(self.dt)),
              update_frequency=(float(self.cfg.lidar.update_frequency_hz)),
              max_range=(float(self.cfg.lidar.far_plane)),
              min_range=(float(self.cfg.lidar.min_range)),
              num_sensors=1,
              horizontal_line_num=(int(self.cfg.lidar.airy_n_azimuth)),
              vertical_line_num=(int(self.cfg.lidar.airy_n_elevation)),
              horizontal_fov_deg_min=(-180.0),
              horizontal_fov_deg_max=180.0,
              vertical_fov_deg_min=(float(self.cfg.lidar.airy_vertical_fov_deg[0])),
              vertical_fov_deg_max=(float(self.cfg.lidar.airy_vertical_fov_deg[1])),
              airy_horizontal_resolution_deg=(float(self.cfg.lidar.airy_horizontal_resolution_deg)),
              return_pointcloud=True,
              pointcloud_in_world_frame=False,
              randomize_placement=False,
              enable_sensor_noise=(bool(self.cfg.lidar.enable_sensor_noise)),
              pixel_std_dev_multiplier=(float(self.cfg.lidar.pixel_std_dev_multiplier)),
              pixel_dropout_prob=(float(self.cfg.lidar.pixel_dropout_prob)),
              random_distance_noise=(float(self.cfg.lidar.random_distance_noise)),
              random_angle_noise=(float(self.cfg.lidar.random_angle_noise)))
            lidar_env = {'device':self.device, 
             'num_envs':self.num_envs, 
             'num_sensors':1, 
             'sensor_pos_tensor':self.sensor_pos_tensor, 
             'sensor_quat_tensor':self.sensor_quat_tensor, 
             'mesh_ids':self.mesh_ids}
            self.lidar_sensor = LidarSensor(lidar_env,
              None, lidar_cfg, num_sensors=1, device=(self.device))
        else:
            self.lidar_sensor = None
        self._lidar_positions_wp = wp.from_torch((self.sensor_pos_tensor.view(self.num_envs, 1, 3)),
          dtype=(wp.vec3))
        self._lidar_quat_wp = wp.from_torch((self.sensor_quat_tensor.view(self.num_envs, 1, 4)),
          dtype=(wp.quat))
        from types import SimpleNamespace
        self._noise_ctx = SimpleNamespace(sensor_cfg=SimpleNamespace(enable_sensor_noise=(bool(self.cfg.lidar.enable_sensor_noise)),
          pixel_std_dev_multiplier=(float(self.cfg.lidar.pixel_std_dev_multiplier)),
          pixel_dropout_prob=(float(self.cfg.lidar.pixel_dropout_prob))),
          far_plane=(float(self.cfg.lidar.far_plane)))
        self._apply_noise = LidarSensor.apply_noise
        offset_pos = list(self.cfg.lidar.offset_pos)
        self._sensor_translation = torch.tensor(offset_pos,
          dtype=(torch.float32), device=(self.device)).view(1, 3).repeat(self.num_envs, 1)
        rpy = list(self.cfg.lidar.sensor_offset_rpy)
        offset_q = quat_from_euler_xyz(torch.tensor((float(rpy[0])), device=(self.device)), torch.tensor((float(rpy[1])), device=(self.device)), torch.tensor((float(rpy[2])), device=(self.device)))
        self._sensor_offset_quat = offset_q.view(1, 4).repeat(self.num_envs, 1)
        from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import load_selected_channels, save_selected_channels, select_ground_grid_channels
        if ea2c.EA2_SELECTED_CHANNELS_FILE.exists():
            self.selected_channels = load_selected_channels()
        else:
            self.selected_channels = select_ground_grid_channels()
            save_selected_channels(self.selected_channels)
        self.range_max = float(self.selected_channels["max_range"])
        self.selected_ray_indices = self.selected_channels["ray_indices"].to(self.device)
        self.selected_ray_directions = self.selected_channels["ray_directions"].to(self.device)
        self.range_image = empty_range_image(self.num_envs, self.range_max, self.device)
        n_sel = ea2c.EA2_RANGE_DIM
        ray_dir_tensor = self.selected_ray_directions.unsqueeze(1).contiguous()
        self._reduced_ray_vectors = wp.from_torch(ray_dir_tensor,
          dtype=(wp.vec3))
        self._reduced_lidar_tensor = torch.zeros((
         self.num_envs, 1, n_sel, 1, 3),
          dtype=(torch.float32),
          device=(self.device))
        self._reduced_dist_tensor = torch.zeros((
         self.num_envs, 1, n_sel, 1),
          dtype=(torch.float32),
          device=(self.device))
        self._reduced_lidar_warp = wp.from_torch((self._reduced_lidar_tensor),
          dtype=(wp.vec3))
        self._reduced_dist_warp = wp.from_torch((self._reduced_dist_tensor),
          dtype=(wp.float32))
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from .path_parallel import init_worker as _pp_init_worker
        self._path_executor = None
        workers = int(getattr(self.cfg.path, "path_plan_workers", 8))
        if workers > 0:
            try:
                self._path_executor = ProcessPoolExecutor(max_workers=workers,
                  mp_context=(multiprocessing.get_context("fork")),
                  initializer=_pp_init_worker,
                  initargs=(
                 self.map_data.occupancy,
                 self.map_data.inflated,
                 self._make_path_cfg()))
            except Exception:
                self._path_executor = None

    def _update_lidar(self):
        """Advance the global 10 Hz clock and refresh the 187-dim range image.

        Two paths are supported:

        * reduced path (training, headless): raycast only the 187 selected
          channels;
        * full path (debug/viewer): raycast the full Airy pattern and extract
          the 187 selected channels, keeping the full point cloud for
          comparison.
        """
        self._lidar_timer += 1
        if self._lidar_timer % self._lidar_decimation != 0:
            return

        import warp as wp
        from isaacgym.torch_utils import quat_apply, quat_mul
        from legged_gym.utils.LidarSensor.sensor_kernels.lidar_kernels_warp import (
            LidarWarpKernels,
        )

        current_q = yaw_quat_from_heading(self.heading)
        self.sensor_quat_tensor.copy_(
            quat_mul(current_q, self._sensor_offset_quat)
        )
        self.sensor_pos_tensor.copy_(
            self.base_pos + quat_apply(current_q, self._sensor_translation)
        )

        use_full = not bool(getattr(self.cfg.lidar, "use_reduced_raycast", True))
        if use_full:
            lidar_points, lidar_dist = self.lidar_sensor.update()
            points = lidar_points.view(self.num_envs, -1, 3)
            dists = lidar_dist.view(self.num_envs, -1)
            n_total = int(points.numel() // 3)
            points_body = quat_apply(
                self._sensor_offset_quat[0:1].expand(n_total, 4),
                points.reshape(-1, 3),
            ).reshape(self.num_envs, -1, 3) + self._sensor_translation.unsqueeze(1)
            fresh = extract_selected_range_image(
                dists, self.selected_ray_indices, self.range_max
            )
            refresh_range_image_from_scan(
                self.range_image, fresh, self.range_image_stale
            )
            if self.viewer is not None and self._debug_env_ids:
                self._debug_is_reduced = False
                ids = torch.tensor(
                    self._debug_env_ids, dtype=torch.long, device=self.device
                )
                self._debug_points = points_body[ids].clone()
                self._debug_dists = dists[ids].clone()
        else:
            wp.launch(
                kernel=(LidarWarpKernels.draw_optimized_kernel_pointcloud),
                dim=(self.num_envs, 1, ea2c.EA2_RANGE_DIM, 1),
                inputs=[
                    self.mesh_ids,
                    self._lidar_positions_wp,
                    self._lidar_quat_wp,
                    self._reduced_ray_vectors,
                    float(self.cfg.lidar.far_plane),
                    self._reduced_lidar_warp,
                    self._reduced_dist_warp,
                    False,
                ],
                device=(self.device),
            )
            lidar_points = wp.to_torch(self._reduced_lidar_warp)
            lidar_dist = wp.to_torch(self._reduced_dist_warp)
            lidar_points, lidar_dist = self._apply_noise(
                self._noise_ctx, lidar_points, lidar_dist
            )
            dists = lidar_dist.view(self.num_envs, -1)
            fresh = build_selected_range_image(dists, self.range_max)
            refresh_range_image_from_scan(
                self.range_image, fresh, self.range_image_stale
            )
        if self.viewer is not None and self._debug_env_ids:
            self._debug_is_reduced = True
            points = lidar_points.view(self.num_envs, -1, 3)
            n_total = int(points.numel() // 3)
            points_body = quat_apply(
                self._sensor_offset_quat[0:1].expand(n_total, 4),
                points.reshape(-1, 3),
            ).reshape(self.num_envs, -1, 3) + self._sensor_translation.unsqueeze(1)
            ids = torch.tensor(
                self._debug_env_ids, dtype=torch.long, device=self.device
            )
            self._debug_points = points_body[ids].clone()
            self._debug_dists = dists[ids].clone()


    def _envelope_debug_points_world(self, env_id: "int") -> "np.ndarray":
        """Return the 6 body-frame envelope vertices in world frame.

        Mirrors ``spider_envelop._envelope_debug_points_world``: local z is
        ``debug_ground_z_offset`` above the world ground plane and only the
        body yaw is applied, so the footprint always sits at z=offset.
        """
        hex_xy = self._compute_hex_world()[env_id].detach().cpu().numpy()
        z = float(self.cfg.envelope.debug_ground_z_offset)
        return np.concatenate([
         hex_xy, np.full((6, 1), z, dtype=(np.float32))],
          axis=1)

    @staticmethod
    def _make_bold_envelope_lines(
        points: np.ndarray,
        edges: Sequence[Tuple[int, int]],
        edge_colors: Sequence[Tuple[float, float, float]],
        line_radius: float,
        line_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Replicate ``spider_envelop._make_bold_envelope_lines``.

        Each edge is drawn as a tube of ``line_samples`` parallel offset lines
        around the edge tangent, making the footprint clearly visible.
        """
        vertices = []
        colors = []
        for (a, b), color in zip(edges, edge_colors):
            p0 = np.asarray(points[a], dtype=np.float32)
            p1 = np.asarray(points[b], dtype=np.float32)
            offsets = [np.zeros(3, dtype=np.float32)]

            edge_vec = p1 - p0
            edge_norm = float(np.linalg.norm(edge_vec))
            if line_radius > 0.0 and line_samples > 1 and edge_norm > 1e-6:
                tangent = edge_vec / edge_norm
                ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                if abs(float(np.dot(tangent, ref))) > 0.9:
                    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                normal_a = np.cross(tangent, ref)
                normal_a = normal_a / max(float(np.linalg.norm(normal_a)), 1e-6)
                normal_b = np.cross(tangent, normal_a)
                normal_b = normal_b / max(float(np.linalg.norm(normal_b)), 1e-6)

                for sample_idx in range(line_samples):
                    angle = 2.0 * np.pi * sample_idx / line_samples
                    offset = line_radius * (
                        np.cos(angle) * normal_a + np.sin(angle) * normal_b
                    )
                    offsets.append(offset.astype(np.float32))

            for offset in offsets:
                vertices.append([p0 + offset, p1 + offset])
                colors.append(color)

        return (
            np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
            np.asarray(colors, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Debug visualization (README 2.7 red/green point cloud)
    # ------------------------------------------------------------------
    def _draw_debug_vis(self) -> "None":
        """Draw red/green LiDAR points and the current envelope footprint.

        Only ``cfg.lidar.debug_env_ids`` are drawn, and only when a viewer
        exists.  The point cloud is the latest *noisy* 10 Hz frame cached by
        :meth:`_update_lidar`; drawing happens after a LiDAR update, so the
        lines appear on the next viewer frame.
        """
        if self.viewer is None or self._debug_points is None:
            return
        from isaacgym import gymapi, gymutil
        from isaacgym.torch_utils import quat_apply
        self.gym.clear_lines(self.viewer)
        far_plane = float(self.cfg.lidar.far_plane)
        stride = max(1, self._debug_point_stride)
        red_geom = gymutil.WireframeSphereGeometry(0.04,
          4, 4, None, color=(1.0, 0.0, 0.0))
        green_geom = gymutil.WireframeSphereGeometry(0.03,
          4, 4, None, color=(0.0, 1.0, 0.0))
        for k, eid in enumerate(self._debug_env_ids):
            if eid >= self.num_envs:
                pass
            else:
                dists = self._debug_dists[k]
                red, green = selected_channel_mask(dists, self.selected_ray_indices, self._debug_is_reduced, far_plane)
                pts_body = self._debug_points[k][::stride]
                red = red[::stride]
                green = green[::stride]
                n = pts_body.shape[0]
                base_q = yaw_quat_from_heading(self.heading[eid:eid + 1]).expand(n, 4)
                pts_world = self.base_pos[eid:eid + 1] + quat_apply(base_q, pts_body)
                pts_world = pts_world.detach().cpu().numpy()
                red_pts = pts_world[red.cpu().numpy()]
                green_pts = pts_world[green.cpu().numpy()]
                for pt in red_pts:
                    pose = gymapi.Transform(gymapi.Vec3(float(pt[0]), float(pt[1]), float(pt[2])))
                    gymutil.draw_lines(red_geom, self.gym, self.viewer, self.envs[eid], pose)
                else:
                    for pt in green_pts:
                        pose = gymapi.Transform(gymapi.Vec3(float(pt[0]), float(pt[1]), float(pt[2])))
                        gymutil.draw_lines(green_geom, self.gym, self.viewer, self.envs[eid], pose)
                    else:
                        points = self._envelope_debug_points_world(eid)
                        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]
                        color = tuple(self.cfg.envelope.debug_color)
                        line_verts, line_colors = self._make_bold_envelope_lines(points, edges, [
                         color] * len(edges), float(self.cfg.envelope.debug_line_radius), int(self.cfg.envelope.debug_line_samples))

                if line_verts.size:
                    self.gym.add_lines(self.viewer, self.envs[eid], int(line_colors.shape[0]), line_verts, line_colors)

    def _sample_start_xy(self) -> np.ndarray:
        """Sample a spawn position inside the 4x4 tile area (exclude 5m border)."""
        spawn = self._spawn_cells
        if spawn.shape[0] == 0:
            raise RuntimeError("map has no spawn cells inside the tile area")
        idx = spawn[self._rng.integers(0, spawn.shape[0])]
        return np.asarray(
            (
                ea2c.EA2_WORLD_MIN_XY
                + (idx[1] + 0.5) * ea2c.EA2_RESOLUTION_M,
                ea2c.EA2_WORLD_MIN_XY
                + (idx[0] + 0.5) * ea2c.EA2_RESOLUTION_M,
            ),
            dtype=np.float64,
        )
    def _sample_goal_xy(self) -> np.ndarray:
        """Sample a goal inside the 4x4 tile area with obstacle clearance.

        Goals use the same ``_spawn_cells`` as starts so the robot does not
        navigate into the 5m outer border / open area.
        """
        free = self._spawn_cells
        if free.shape[0] == 0:
            raise RuntimeError("map has no goal cells inside the tile area")
        goal_clearance = float(self.cfg.path.goal_min_obstacle_dist)
        for _ in range(200):
            goal_idx = free[self._rng.integers(0, free.shape[0])]
            goal_xy = np.asarray(
                (
                    ea2c.EA2_WORLD_MIN_XY
                    + (goal_idx[1] + 0.5) * ea2c.EA2_RESOLUTION_M,
                    ea2c.EA2_WORLD_MIN_XY
                    + (goal_idx[0] + 0.5) * ea2c.EA2_RESOLUTION_M,
                ),
                dtype=np.float64,
            )
            if self._min_obstacle_distance_world(goal_xy) >= goal_clearance:
                return goal_xy
        raise RuntimeError("could not sample a valid goal in 200 attempts")
    def _sample_free_start_goal(self) -> "Tuple[np.ndarray, np.ndarray]":
        """Sample a start/goal pair from the largest safe connected component."""
        return (
         self._sample_start_xy(), self._sample_goal_xy())

    def _min_obstacle_distance_world(self, xy) -> float:
        """Return distance to the nearest obstacle using the precomputed field.

        The map's unsigned distance field is computed once at generation time,
        so this is an O(1) lookup instead of scanning all occupied cells.
        """
        res = ea2c.EA2_RESOLUTION_M
        wmin = ea2c.EA2_WORLD_MIN_XY
        ix = int(np.floor(float(xy[0]) - wmin) / res)
        iy = int(np.floor(float(xy[1]) - wmin) / res)
        if (
            ix < 0
            or ix >= self.occupancy.shape[1]
            or iy < 0
            or iy >= self.occupancy.shape[0]
        ):
            return float("inf")
        return float(self.map_data.distance_field[iy, ix])

    def _make_path_cfg(self) -> "PathCfg":
        """Build the current :class:`PathCfg` from the environment config."""
        return PathCfg(speed_range=(tuple(self.cfg.path.speed_range)),
          resample_time_s=(float(self.cfg.path.resample_time_s)),
          delta_target_deg_range=(tuple(self.cfg.path.delta_target_deg_range)),
          omega_max=(float(self.cfg.path.omega_max)),
          k_p=(float(self.cfg.path.k_p)),
          min_turn_radius=(float(self.cfg.path.min_turn_radius)),
          resample_dist=(float(self.cfg.path.resample_dist)),
          goal_min_obstacle_dist=(float(self.cfg.path.goal_min_obstacle_dist)),
          min_path_len=(float(self.cfg.path.min_path_len)),
          noise_amp_range=(tuple(self.cfg.path.noise_amp_range)),
          noise_fc_hz=(float(self.cfg.path.noise_fc_hz)),
          noise_retries=(int(self.cfg.path.noise_retries)))

    def _sample_new_path(self) -> PathData:
        """Sample a feasible start/goal and plan one noisy A* path."""
        path_cfg = self._make_path_cfg()
        last_err = None
        for _ in range(40):
            try:
                start_xy, goal_xy = self._sample_free_start_goal()
                return plan_path(
                    self.occupancy,
                    self.inflated,
                    start_xy,
                    goal_xy,
                    path_cfg,
                    self._rng,
                )
            except (ValueError, RuntimeError) as err:
                last_err = err
        raise RuntimeError(f"failed to plan a path after 40 attempts: {last_err}")

    def _install_path(self, env_id: "int", path: "Optional[PathData]") -> "None":
        """Install a path into both the Python list and the device batch."""
        self.paths[env_id] = path
        self._path_batch.install(env_id, path)

    def _replan_from_current(self, env_id) -> None:
        """Plan a new path from the current position to a new random goal.

        Called when the robot reaches a goal.  The robot keeps its current
        pose and naturally turns toward the new path tangent on the next
        control step; no episode/GRU reset happens here.
        """
        start_xy = np.asarray(self.paths[env_id].points[-1], dtype=np.float64)
        path_cfg = self._make_path_cfg()
        for _ in range(40):
            try:
                goal_xy = self._sample_goal_xy()
                path = plan_path(
                    self.occupancy,
                    self.inflated,
                    start_xy,
                    goal_xy,
                    path_cfg,
                    self._rng,
                )
                self._install_path(env_id, path)
                self.s[env_id] = 0.0
                if path.segment_dirs is not None and len(path.segment_dirs) > 0:
                    first_dir = float(path.segment_dirs[0])
                else:
                    first_dir = float(path.yaws[0])
                self.tangent[env_id] = first_dir
                self.tangent_rate[env_id] = 0.0
                self._turn_target[env_id] = first_dir
                self._turn_in_place[env_id] = True
                return
            except (ValueError, RuntimeError):
                continue
        self._replan_reset_buf[env_id] = True

    def _apply_replanned_path(self, env_id: "int", path: "PathData") -> "None":
        """Install a newly planned path and enter turn-in-place."""
        self._install_path(env_id, path)
        self.s[env_id] = 0.0
        first_dir = float(path.segment_dirs[0]) if (path.segment_dirs is not None and len(path.segment_dirs) > 0) else (float(path.yaws[0]))
        self.tangent[env_id] = first_dir
        self.tangent_rate[env_id] = 0.0
        self._turn_target[env_id] = first_dir
        self._turn_in_place[env_id] = True

    def _plan_paths_parallel(self, env_ids) -> "dict":
        """Plan new reset paths for a batch of envs, parallel when beneficial."""
        ids = [int(e) for e in env_ids]
        threshold = int(getattr(self.cfg.path, "path_plan_batch_threshold", 4))
        if self._path_executor is None or len(ids) < threshold:
            return {env_id: self._sample_new_path() for env_id in ids}
        from concurrent.futures import as_completed
        from .path_parallel import plan_path_task
        future_map = {}
        for env_id in ids:
            start_xy, goal_xy = self._sample_free_start_goal()
            seed = int(self._rng.integers(0, 2147483647))
            future = self._path_executor.submit(plan_path_task, seed, start_xy, goal_xy)
            future_map[future] = env_id
        else:
            results = {}
            for future in as_completed(future_map):
                env_id = future_map[future]
                try:
                    results[env_id] = future.result()
                except Exception:
                    results[env_id] = self._sample_new_path()

            else:
                return results

    def _batch_replan(self, env_ids) -> "None":
        """Replan a batch of reached-goal envs, parallel when beneficial."""
        ids = [int(e) for e in env_ids]
        threshold = int(getattr(self.cfg.path, "path_plan_batch_threshold", 4))
        if self._path_executor is None or len(ids) < threshold:
            for env_id in ids:
                self._replan_from_current(env_id)
            else:
                return

        from concurrent.futures import as_completed
        from .path_parallel import plan_path_task
        future_map = {}
        for env_id in ids:
            start_xy = np.asarray((self.paths[env_id].points[-1]),
              dtype=(np.float64))
            goal_xy = self._sample_goal_xy()
            seed = int(self._rng.integers(0, 2147483647))
            future = self._path_executor.submit(plan_path_task, seed, start_xy, goal_xy)
            future_map[future] = env_id
        else:
            for future in as_completed(future_map):
                env_id = future_map[future]
                try:
                    path = future.result()
                    self._apply_replanned_path(env_id, path)
                except Exception:
                    self._replan_from_current(env_id)

    def _reset_one_env(self, env_id: "int", path: "Optional[PathData]"=None):
        if path is None:
            path = self._sample_new_path()
        self._install_path(env_id, path)
        start_xy = path.points[0]
        start_yaw = float(path.segment_dirs[0]) if (path.segment_dirs is not None and len(path.segment_dirs) > 0) else (float(path.yaws[0]))
        self.s[env_id] = 0.0
        self.base_pos[(env_id, 0)] = float(start_xy[0])
        self.base_pos[(env_id, 1)] = float(start_xy[1])
        self.heading[env_id] = start_yaw
        self.tangent[env_id] = start_yaw
        self.tangent_rate[env_id] = 0.0
        self.delta_actual[env_id] = 0.0
        self.omega[env_id] = 0.0
        self.v[env_id] = float(self.cfg.path.speed_range[0])
        self.delta_target[env_id] = math.radians(float(self.cfg.path.delta_target_deg_range[0]))
        self.height_target[env_id] = float(self.cfg.height.min_m)
        self.height_filter[env_id] = float(self.height_target[env_id])
        self.base_pos[(env_id, 2)] = float(self.height_target[env_id])
        self.height_wobble_state[env_id] = 0.0
        self.pos_sway_state[env_id] = 0.0
        self.heading_sway_state[env_id] = 0.0
        self.pos_sway_amp[env_id] = float(self.cfg.sway.pos_amp_range[0])
        self.heading_sway_amp[env_id] = float(self.cfg.sway.heading_amp_range[0])
        self.height_wobble_amp[env_id] = float(self.cfg.height.wobble_amp_range[0])
        self.episode_length_buf[env_id] = 0
        self._turn_in_place[env_id] = False
        self._turn_target[env_id] = 0.0
        self._replan_reset_buf[env_id] = False
        self.reset_buf[env_id] = 1
        zero = torch.zeros_like(self.actions_mapped[env_id])
        self.actions_mapped[env_id], self.actions_target[env_id] = map_actions_to_params_with_target(zero, self._envelope_low_dev, self._envelope_high_dev, float(self.cfg.envelope.soft_dof_pos_limit), float(self.cfg.envelope.action_max))
        self.actions[env_id] = 0.0
        self.last_actions_raw[env_id] = 0.0
        self.range_image_stale[env_id] = True
        self.range_image[env_id] = self.range_max
        if self._oracle_smoother is not None:
            self._oracle_smoother.reset_ids([env_id])

    def _log_segment(self, env_ids):
        """Log accumulated reward sums for a set of envs and reset the sums.

        This is used at full environment resets (including the 30s timeout
        reset), independently of whether the GRU memory is cleared.
        """
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(self.device)
        ep = {}
        for name in self.reward_scales.keys():
            ep[f"rew_{name}"] = self.episode_sums[name][env_ids].mean() / max(self.max_episode_length, 1)
        else:
            for name, buf in self.episode_metrics.items():
                ep[f"ep_{name}"] = buf[env_ids].mean() / max(self.max_episode_length, 1)
            else:
                self.extras["episode"] = ep
                for name in self.reward_scales.keys():
                    self.episode_sums[name][env_ids] = 0.0
                else:
                    for buf in self.episode_metrics.values():
                        buf[env_ids] = 0.0

    def reset_idx(self, env_ids):
        """Reset selected envs and log episode metrics."""
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(self.device)
        ids = env_ids.tolist()
        self._log_segment(env_ids)
        preplanned = self._plan_paths_parallel(ids)
        for env_id in ids:
            self._reset_one_env(env_id, path=(preplanned.get(env_id)))
        else:
            for i in ids:
                vx, vy, omega_out = ego_motion(self.v[i:i + 1], self.heading[i:i + 1], self.tangent[i:i + 1], self.omega[i:i + 1])
                self.ego_motion[(i, 0)] = vx
                self.ego_motion[(i, 1)] = vy
                self.ego_motion[(i, 2)] = omega_out
            else:
                if self.cfg.env.send_timeouts:
                    self.extras["time_outs"] = self.time_out_buf

    def _step_kinematics(self):
        """Advance each env along its reference path (batched backend)."""
        self._step_kinematics_batched()

    def _step_kinematics_batched(self):
        """Batched equivalent of :meth:`_step_kinematics` (stage 1)."""
        n = self.num_envs
        dt = self.dt
        k_p = float(self.cfg.path.k_p)
        omega_max = float(self.cfg.path.omega_max)
        height_min = float(self.cfg.height.min_m)
        valid = self._path_batch.valid
        was_turn = self._turn_in_place & valid
        zero_rate = torch.zeros(n, dtype=(torch.float32), device=(self.device))
        arange = torch.arange(n, device=(self.device))
        if bool(was_turn.any()):
            tg = self._turn_target
            delta = wrap_to_pi(self.heading - tg)
            omega_cmd = k_p * wrap_to_pi(self.delta_target - delta)
            omega = torch.clamp(omega_cmd, -omega_max, omega_max)
            heading_new = wrap_to_pi(self.heading + omega * dt)
            delta_new = wrap_to_pi(heading_new - tg)
            aligned = delta_new.abs() < 0.05
            self.heading.copy_(torch.where(was_turn, heading_new, self.heading))
            self.omega.copy_(torch.where(was_turn, omega, self.omega))
            self.delta_actual.copy_(torch.where(was_turn, delta_new, self.delta_actual))
            self.tangent.copy_(torch.where(was_turn, tg, self.tangent))
            self.tangent_rate.copy_(torch.where(was_turn, zero_rate, self.tangent_rate))
            self._turn_in_place.copy_(was_turn & ~aligned)
            self.ego_motion[(was_turn, 0)] = 0.0
            self.ego_motion[(was_turn, 1)] = 0.0
            self.ego_motion[(was_turn, 2)] = omega[was_turn]
        active = valid & ~was_turn
        q_old = self._path_batch.query(self.s)
        near_corner = active & q_old.has_next_corner & (self.s + self.v * dt >= q_old.next_corner)
        advance = active & ~near_corner
        s_after = self.s + self.v * dt
        last_idx = (self._path_batch.lengths - 1).clamp(0, self._path_batch.max_points - 1)
        last = self._path_batch.arc[(arange, last_idx)]
        reached = advance & (s_after >= last)
        interp = advance & ~reached
        s_corner = torch.minimum(q_old.next_corner + 0.0001, last)
        sample_s = torch.where(near_corner, s_corner, s_after)
        q = self._path_batch.query(sample_s)
        self.s.copy_(torch.where(near_corner, s_corner, torch.where(advance, s_after, self.s)))
        sample = interp | near_corner
        if bool(sample.any()):
            self.base_pos[(sample, 0)] = q.xy[(sample, 0)]
            self.base_pos[(sample, 1)] = q.xy[(sample, 1)]
            self.base_pos[(sample, 2)] = height_min
        if bool(interp.any()):
            tang = q.tangent
            delta = wrap_to_pi(self.heading - tang)
            omega_cmd = k_p * wrap_to_pi(self.delta_target - delta)
            omega = torch.clamp(omega_cmd, -omega_max, omega_max)
            heading_new = wrap_to_pi(self.heading + omega * dt)
            delta_new = wrap_to_pi(heading_new - tang)
            self.heading.copy_(torch.where(interp, heading_new, self.heading))
            self.omega.copy_(torch.where(interp, omega, self.omega))
            self.delta_actual.copy_(torch.where(interp, delta_new, self.delta_actual))
            self.tangent.copy_(torch.where(interp, tang, self.tangent))
            self.tangent_rate.copy_(torch.where(interp, zero_rate, self.tangent_rate))
            vx = self.v * torch.cos(delta_new)
            vy = self.v * torch.sin(delta_new)
            self.ego_motion[(interp, 0)] = vx[interp]
            self.ego_motion[(interp, 1)] = vy[interp]
            self.ego_motion[(interp, 2)] = omega[interp]
        if bool(near_corner.any()):
            target = q_old.next_target
            not_aligned = wrap_to_pi(self.heading - target).abs() >= 0.05
            self._turn_target.copy_(torch.where(near_corner, target, self._turn_target))
            self.tangent.copy_(torch.where(near_corner, target, self.tangent))
            self.tangent_rate.copy_(torch.where(near_corner, zero_rate, self.tangent_rate))
            self.ego_motion[(near_corner, 0)] = 0.0
            self.ego_motion[(near_corner, 1)] = 0.0
            self.ego_motion[(near_corner, 2)] = 0.0
            self._turn_in_place.copy_(self._turn_in_place | near_corner & not_aligned)
        if bool(reached.any()):
            pending = reached.nonzero(as_tuple=False).flatten().tolist()
            self._batch_replan(pending)

    def _compute_hex_world(self) -> "torch.Tensor":
        """Return world-frame offset hexagon vertices for all envs (CPU)."""
        params = self.actions_mapped.to("cpu")
        body_hex = compute_hex_vertices(params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4])
        heading = self.heading.to("cpu")
        base_xy = self.base_pos[:, :2].to("cpu")
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        rot = torch.stack([
         torch.stack([cos_h, -sin_h], dim=(-1)),
         torch.stack([sin_h, cos_h], dim=(-1))],
          dim=(-2))
        world = torch.einsum("eij,evj->evi", rot, body_hex) + base_xy.unsqueeze(1)
        return world

    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        low = self._envelope_low_dev
        high = self._envelope_high_dev
        potential = potential_reward(self.actions_mapped, low, high)
        violation, self._collision_hard = hex_collision_terms((self.actions_mapped),
          (self.heading),
          (self.base_pos[:, :2]),
          (self.distance_field),
          margin=(float(self.cfg.envelope.margin)),
          soft_margin=(float(self.cfg.envelope.soft_margin)))
        act_rate = raw_action_rate_term(self.actions, self.last_actions_raw)
        limit_violation = envelope_limit_violation(self.actions_target, low, high, float(self.cfg.envelope.soft_dof_pos_limit))
        oracle_params, oracle_hard_raw = compute_direct_oracle_params_with_stats((self.heading),
          (self.base_pos[:, :2]),
          (self.distance_field),
          low,
          high,
          margin=(float(self.cfg.envelope.oracle_margin)),
          step=(float(self.cfg.envelope.oracle_step)),
          max_dist=(float(self.cfg.envelope.oracle_max_dist)),
          soft_dof_pos_limit=(float(self.cfg.envelope.soft_dof_pos_limit)),
          interp_crossing=(bool(getattr(self.cfg.envelope, "oracle_interp_crossing", True))))
        # Rate-limited smoothing: the supervised target (and the telemetry that
        # describes it) is the final candidate; the smoother is advanced
        # exactly once per control step here.  Disabled/absent -> identity.
        smoother = getattr(self, "_oracle_smoother", None)
        if smoother is not None:
            oracle_params = smoother.update(oracle_params)
        oracle_mse = ((normalized_envelope_params(self.actions_mapped, low, high) - normalized_envelope_params(oracle_params, low, high)) ** 2).mean(dim=(-1))
        _, oracle_hard = hex_collision_terms(oracle_params,
          (self.heading),
          (self.base_pos[:, :2]),
          (self.distance_field),
          margin=(float(self.cfg.envelope.margin)),
          soft_margin=(float(self.cfg.envelope.soft_margin)))
        terms = {
         'potential': potential, 
         'collision': violation, 
         'action_rate': act_rate, 
         'envelope_limits': limit_violation, 
         'oracle_mse': oracle_mse}
        for name, scale in self.reward_scales.items():
            rew = terms[name] * scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
        else:
            self.episode_metrics["collision_hard_max"] += self._collision_hard
            self.episode_metrics["raw_action_abs_mean"] += self.actions.abs().mean(dim=(-1))
            self.episode_metrics["envelope_limits_active_ratio"] += (limit_violation > 0).to(torch.float32)
            self.episode_metrics["oracle_unsafe_before_ratio"] += (oracle_hard_raw > 0).to(torch.float32)
            self.episode_metrics["oracle_unsafe_ratio"] += (oracle_hard > 0).to(torch.float32)
            self.episode_metrics["oracle_potential"] += potential_reward(oracle_params, low, high)
            oracle_sq = (
                normalized_envelope_params(self.actions_mapped, low, high)
                - normalized_envelope_params(oracle_params, low, high)
            ) ** 2
            for j, metric_name in enumerate((
                "oracle_mse_front_width",
                "oracle_mse_middle_width",
                "oracle_mse_back_width",
                "oracle_mse_forward_limit",
                "oracle_mse_backward_limit",
            )):
                self.episode_metrics[metric_name] += oracle_sq[:, j]
        if smoother is not None:
            self.episode_metrics["oracle_snap_ratio"] += smoother.snapped.to(
                torch.float32
            )

    def _compute_observations(self):
        self.obs_buf[:] = assemble_observation((self.range_image),
          (self.ego_motion),
          max_range=(self.range_max))

    def compute_observations(self):
        """Alias for :meth:`_compute_observations` (LeggedRobot-style API)."""
        self._compute_observations()

    def compute_reward(self):
        """Alias for :meth:`_compute_rewards` (LeggedRobot-style API)."""
        self._compute_rewards()

    def step(self, actions):
        """Old legged_gym 5-tuple step interface (README 2.9)."""
        clip_actions = float(self.cfg.normalization.clip_actions)
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.actions_mapped, self.actions_target = map_actions_to_params_with_target(self.actions, self._envelope_low_dev, self._envelope_high_dev, float(self.cfg.envelope.soft_dof_pos_limit), float(self.cfg.envelope.action_max))
        self.extras.pop("episode", None)
        self.render()
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._step_kinematics()
        self._update_lidar()
        self.time_out_buf[:] = False
        self._compute_rewards()
        hard_reset = self._replan_reset_buf.clone()
        self._replan_reset_buf[:] = False
        timeout = self.episode_length_buf >= self.max_episode_length
        timeout_ids = timeout.nonzero(as_tuple=False).flatten()
        reset_ids = (hard_reset | timeout).nonzero(as_tuple=False).flatten()
        self.reset_buf[:] = 0
        self.time_out_buf[:] = False
        if len(reset_ids) > 0:
            self.reset_idx(reset_ids)
            self.reset_buf[reset_ids] = 1
            self.time_out_buf[timeout_ids] = True
        self._compute_observations()
        self.last_actions_raw[:] = self.actions[:]
        clip_obs = float(self.cfg.normalization.clip_observations)
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self._draw_debug_vis()
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return (self.obs_buf,
         self.privileged_obs_buf,
         self.rew_buf,
         self.reset_buf,
         self.extras)
