"""M1 envelope_adaptive_2 environment (``EL_4090_EA2``).

This module integrates T1-T6 (map, path, hex geometry, range image, Airy
mount, LiDAR noise) into a simplified legged_gym ``BaseTask`` with no robot
actor.  The policy sees a 453-dim observation (450 range-image + 3 ego-motion)
and emits 5 raw envelope parameters which are mapped in the environment to the
spider_envelop ranges.

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
    collision_cell_ratio,
    compute_hex_vertices,
    envelope_params_to_condition,
    offset_hexagon,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.map_generator import (
    MapGenCfg,
    generate_map,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.path_planner import (
    PathCfg,
    PathData,
    plan_path,
    wrap_to_pi as _np_wrap_to_pi,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.range_image import (
    aggregate_range_image,
    range_image_observation,
)
from legged_gym.utils.envelop.network.haa_swing_range import (
    EnvelopeConditionSpec,
    load_envelope_condition_spec,
)

# ---------------------------------------------------------------------------
# Module-level pure helpers (Isaac-free, unit-testable)
# ---------------------------------------------------------------------------


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle/tensor to ``[-pi, pi)`` using PyTorch semantics."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


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


def map_actions_to_params(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Map raw policy actions through sigmoid+affine to envelope parameters.

    ``actions`` shape ``(..., 5)``, ``low``/``high`` shape ``(5,)``.  This is
    the environment-side mapping required by README 2.5/2.6.
    """
    norm = torch.sigmoid(actions)
    return low + norm * (high - low)


def envelope_condition_from_params(
    params5: torch.Tensor,
    spec: EnvelopeConditionSpec,
) -> torch.Tensor:
    """Convert 5 mapped envelope params to the frozen 8-dim condition."""
    return envelope_params_to_condition(params5, spec)


def potential_reward(
    params5: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Normalized-parameter mean potential (README 2.8).

    ``backward_limit`` is negated before normalization, matching the physical
    direction used by ``apply_env_morphology_priors``.
    """
    low5 = low[:5]
    high5 = high[:5]
    norm = (params5 - low5) / (high5 - low5).clamp_min(1e-6)
    # backward_limit: value_low = -high, value_high = -low.
    norm = norm.clone()
    norm[..., 4] = (-params5[..., 4] - (-high5[4])) / (
        (-low5[4]) - (-high5[4])
    ).clamp_min(1e-6)
    return norm.clamp(0.0, 1.0).mean(dim=-1)


def collision_ratio(
    hex_vertices_world_xy: torch.Tensor,
    occupancy,
    margin: float = 0.05,
) -> torch.Tensor:
    """Collision ratio of a hexagon against raw occupancy.

    The input is the world-frame hexagon vertices.  When ``margin > 0`` the
    exact half-plane offset is applied before counting covered occupied cells.
    The returned value is a positive ratio in ``[0, 1]``; the environment
    multiplies it by the (negative) collision reward scale.
    """
    if margin > 0.0:
        hex_vertices_world_xy = offset_hexagon(hex_vertices_world_xy, margin)
    return collision_cell_ratio(hex_vertices_world_xy, occupancy)


def action_rate_term(
    actions_mapped: torch.Tensor,
    last_actions_mapped: torch.Tensor,
) -> torch.Tensor:
    """Mean squared change of mapped envelope params (README 2.8)."""
    return ((actions_mapped - last_actions_mapped) ** 2).mean(dim=-1)


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


def point_cloud_debug_masks(
    dists: torch.Tensor,
    mapping: torch.Tensor,
    far_plane: float,
    max_range: float,
    r_min: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classify rays for README 2.7 red/green visualization.

    Args:
        dists: Noisy ray distances, shape ``(..., N)``.
        mapping: Flat Airy mapping table, shape ``(N,)``.
        far_plane: ``LidarConfig.max_range`` (60 m); ``d >= far_plane`` is a
            no-hit ray.
        max_range/r_min: Effective 450-bucket range ``[r_min, max_range]``.

    Returns:
        ``(red, green)`` boolean masks with the same shape as ``dists``.
        Red = real hit whose channel is in the mapping table and whose noisy
        distance participates in the 450-dim aggregation.  Green = other real
        hits.  No-hit and dropout rays are excluded from both masks.
    """
    far = float(far_plane)
    max_r = float(max_range)
    min_r = float(r_min)
    hit = dists < (far - 1e-3)
    mapped = mapping.to(device=dists.device) >= 0
    in_agg_range = (dists >= min_r) & (dists <= max_r)
    red = hit & mapped & in_agg_range
    green = hit & ~red
    return red, green



def sway_position_acceptable(
    prev_xy: Sequence[float],
    cand_xy: Sequence[float],
    inflated: np.ndarray,
) -> bool:
    """Check that a sway candidate is free and reachable from the old pose.

    Uses the same inflated-grid point/segment checks as the path planner.
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2.path_planner import (
        _point_free,
        _segment_clear,
    )

    return bool(_point_free(cand_xy, inflated)) and bool(
        _segment_clear(prev_xy, cand_xy, inflated)
    )


def interpolate_path(
    path: PathData,
    s: float,
) -> Tuple[np.ndarray, float, float]:
    """Interpolate a :class:`PathData` at arc length ``s``.

    Returns ``(xy, tangent_yaw, tangent_rate)`` where ``tangent_rate`` is the
    discrete curvature ``wrap_to_pi(dyaw) / ds`` in rad/s?  Actually rad/m;
    callers multiply by ``v`` to obtain rad/s.  For endpoints the curvature is
    zero.
    """
    points = np.asarray(path.points, dtype=np.float64)
    yaws = np.asarray(path.yaws, dtype=np.float64)
    arc = np.asarray(path.arc, dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        raise ValueError("empty path")
    if n == 1 or s <= arc[0] + 1e-12:
        return points[0].copy(), float(yaws[0]), 0.0
    if s >= arc[-1] - 1e-12:
        return points[-1].copy(), float(yaws[-1]), 0.0

    idx = int(np.searchsorted(arc, s, side="right") - 1)
    idx = min(max(idx, 0), n - 2)
    ds = float(arc[idx + 1] - arc[idx])
    if ds <= 1e-12:
        return points[idx].copy(), float(yaws[idx]), 0.0
    t = float((s - arc[idx]) / ds)
    xy = (1.0 - t) * points[idx] + t * points[idx + 1]
    tangent = float(yaws[idx])
    tangent_rate = float(_np_wrap_to_pi(yaws[idx + 1] - yaws[idx])) / ds
    return xy, tangent, tangent_rate


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class EL_4090_EA2(BaseTask):
    """Simplified BaseTask for envelope perception (M1, no robot actor)."""

    def __init__(
        self,
        cfg: El4090EA2Cfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
        task_name: str = "el4090_ea2",
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.task_name = task_name

        # M1 has no control decimation: 50 Hz policy step == sim dt.
        self.dt = float(cfg.sim.dt)
        self.max_episode_length = int(
            math.ceil(float(cfg.env.episode_length_s) / self.dt)
        )

        self._spec = load_envelope_condition_spec(ea2c.ENVELOPE_SPEC_CONFIG_PATH)
        self._envelope_low = torch.tensor(
            self._spec.low[:5], dtype=torch.float32
        )
        self._envelope_high = torch.tensor(
            self._spec.high[:5], dtype=torch.float32
        )

        self.reward_scales = {
            "potential": float(cfg.rewards.scales.potential),
            "collision": float(cfg.rewards.scales.collision),
            "action_rate": float(cfg.rewards.scales.action_rate),
        }

        # BaseTask creates the sim/viewer and the basic obs/rew buffers.
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._init_buffers()
        self._init_lidar()

    # ------------------------------------------------------------------
    # Simulation creation
    # ------------------------------------------------------------------

    def create_sim(self):
        """Create Isaac sim, a ground plane, empty shared-origin envs and
        static visual obstacle actors (only in env 0)."""
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        # One fixed global map (README 2.2.1).  The map is generated once here
        # and never rebuilt during training.
        map_cfg = MapGenCfg(
            size_m=float(self.cfg.map.size_m),
            resolution_m=float(self.cfg.map.resolution_m),
            grid_shape=tuple(self.cfg.map.grid_shape),
            boundary_occupied=bool(self.cfg.map.boundary_occupied),
            ground_margin_m=float(self.cfg.map.ground_margin_m),
            inflation_m=float(self.cfg.map.inflation_m),
            inflation_cells=int(self.cfg.map.inflation_cells),
            max_gen_attempts=int(self.cfg.map.max_gen_attempts),
            n_validation_paths=int(self.cfg.map.n_validation_paths),
            min_solved_ratio=float(self.cfg.map.min_solved_ratio),
            path_near_obstacle_ratio=float(self.cfg.map.path_near_obstacle_ratio),
            near_obstacle_range=tuple(self.cfg.map.near_obstacle_range),
            require_constraint_primitive=bool(
                self.cfg.map.require_constraint_primitive
            ),
        )
        map_seed = int(getattr(self.cfg, "seed", 42))
        self.map_data = generate_map(map_cfg, seed=map_seed)
        self.occupancy = self.map_data.occupancy
        self.inflated = self.map_data.inflated

        # Physics ground for Isaac (visual/ground only; Warp mesh is the
        # authoritative raycast geometry).
        plane_params = self.gymapi().PlaneParams()
        plane_params.normal = self.gymapi().Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # All envs share the same world origin; bounds are deliberately large
        # enough to cover the single 12x12 map.
        env_lower = self.gymapi().Vec3(-20.0, -20.0, -20.0)
        env_upper = self.gymapi().Vec3(20.0, 20.0, 20.0)
        num_per_row = max(1, int(math.isqrt(self.num_envs)))
        self.envs = []
        self.env_origins = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row
            )
            self.envs.append(env_handle)
            # Actors must be added before creating the next env in Isaac Gym.
            if i == 0:
                self._add_visual_obstacles(env_index=0)

    def gymapi(self):
        """Lazily return the Isaac Gym Python module (kept as a method so the
        pure-helper tests never need Isaac unless the class is instantiated)."""
        import isaacgym

        return isaacgym.gymapi

    def _add_visual_obstacles(self, env_index: int = 0):
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
            asset = self.gym.create_box(
                self.sim, sx, sy, sz, asset_options
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                float(rect.center[0]), float(rect.center[1]), sz / 2.0
            )
            pose.r = gymapi.Quat.from_euler_zyx(float(rect.yaw), 0.0, 0.0)
            self.gym.create_actor(
                env_handle, asset, pose, "ea2_visual_wall", env_index, 0, 0
            )

        for pillar in self.map_data.pillars:
            side = 2.0 * float(pillar.radius)
            sz = float(pillar.height)
            asset = self.gym.create_box(
                self.sim, side, side, sz, asset_options
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                float(pillar.center[0]), float(pillar.center[1]), sz / 2.0
            )
            self.gym.create_actor(
                env_handle, asset, pose, "ea2_visual_pillar", env_index, 0, 0
            )

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _init_buffers(self):
        device = self.device
        n = self.num_envs

        self._lidar_decimation = max(
            1, int(round(1.0 / (self.dt * float(self.cfg.lidar.update_frequency_hz))))
        )
        # Legacy clock: start just before the first decimation boundary so the
        # first policy step performs a global LiDAR scan.
        self._lidar_timer = self._lidar_decimation - 1
        self.common_step_counter = 0

        # Paths / kinematics
        self.paths: List[Optional[PathData]] = [None] * n
        self.s = torch.zeros(n, dtype=torch.float32, device=device)
        self.v = torch.zeros(n, dtype=torch.float32, device=device)
        self.heading = torch.zeros(n, dtype=torch.float32, device=device)
        self.tangent = torch.zeros(n, dtype=torch.float32, device=device)
        self.tangent_rate = torch.zeros(n, dtype=torch.float32, device=device)
        self.delta_target = torch.zeros(n, dtype=torch.float32, device=device)
        self.delta_actual = torch.zeros(n, dtype=torch.float32, device=device)
        self.omega = torch.zeros(n, dtype=torch.float32, device=device)
        self.base_pos = torch.zeros(n, 3, dtype=torch.float32, device=device)
        self.base_quat = torch.zeros(n, 4, dtype=torch.float32, device=device)
        self.base_quat[:, 3] = 1.0

        # Height / sway AR states
        self.height_filter = torch.zeros(n, dtype=torch.float32, device=device)
        self.height_target = torch.zeros(n, dtype=torch.float32, device=device)
        self.height_wobble_state = torch.zeros(
            n, dtype=torch.float32, device=device
        )
        self.pos_sway_state = torch.zeros(n, dtype=torch.float32, device=device)
        self.heading_sway_state = torch.zeros(
            n, dtype=torch.float32, device=device
        )
        self.pos_sway_amp = torch.zeros(n, dtype=torch.float32, device=device)
        self.heading_sway_amp = torch.zeros(
            n, dtype=torch.float32, device=device
        )
        self.height_wobble_amp = torch.zeros(
            n, dtype=torch.float32, device=device
        )

        # Actions / envelope
        self.actions = torch.zeros(
            n, self.num_actions, dtype=torch.float32, device=device
        )
        self.last_actions = torch.zeros(
            n, self.num_actions, dtype=torch.float32, device=device
        )
        self.actions_mapped = torch.zeros(
            n, self.num_actions, dtype=torch.float32, device=device
        )
        self.condition = torch.zeros(
            n, 8, dtype=torch.float32, device=device
        )

        # LiDAR / observations
        self.range_image = empty_range_image(n, ea2c.EA2_RANGE_MAX_M, device)
        self.range_image_stale = torch.ones(n, dtype=torch.bool, device=device)
        self.sensor_pos_tensor = torch.zeros(
            n, 3, dtype=torch.float32, device=device
        )
        self.sensor_quat_tensor = torch.zeros(
            n, 4, dtype=torch.float32, device=device
        )
        self.ego_motion = torch.zeros(
            n, 3, dtype=torch.float32, device=device
        )

        # Debug point-cloud visualization state (populated only when a viewer
        # exists; stored only for cfg.lidar.debug_env_ids).
        self._debug_env_ids = [
            int(i) for i in getattr(self.cfg.lidar, "debug_env_ids", [0])
        ]
        self._debug_point_stride = int(
            getattr(self.cfg.lidar, "debug_point_stride", 1)
        )
        self._debug_points = None
        self._debug_dists = None
        self._debug_base_pos = None
        self._debug_base_quat = None

        # Episode reward sums (scaled, matching LeggedRobot logging)
        self.episode_sums = {
            name: torch.zeros(n, dtype=torch.float32, device=device)
            for name in self.reward_scales.keys()
        }

        self._rng = np.random.default_rng(
            int(getattr(self.cfg, "seed", 42)) + 1
        )

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------

    def _init_lidar(self):
        """Initialize the Warp mesh and the shared LidarSensor instance."""
        import warp as wp
        from isaacgym.torch_utils import (
            quat_apply,
            quat_from_euler_xyz,
            quat_mul,
        )
        from legged_gym.utils.LidarSensor import (
            LidarConfig,
            LidarSensor,
            LidarType,
        )

        wp.init()
        vertices = torch.from_numpy(self.map_data.vertices).to(self.device)
        triangles = np.asarray(self.map_data.triangles, dtype=np.int32).flatten()
        self._wp_mesh = wp.Mesh(
            points=wp.from_torch(vertices, dtype=wp.vec3),
            indices=wp.from_numpy(triangles, dtype=wp.int32, device=self.device),
        )
        self.mesh_ids = wp.array(
            [self._wp_mesh.id], dtype=wp.uint64, device=self.device
        )

        lidar_cfg = LidarConfig(
            sensor_type=LidarType.AIRY,
            dt=float(self.dt),
            update_frequency=float(self.cfg.lidar.update_frequency_hz),
            max_range=float(self.cfg.lidar.far_plane),
            min_range=float(self.cfg.lidar.min_range),
            num_sensors=1,
            horizontal_line_num=int(self.cfg.lidar.airy_n_azimuth),
            vertical_line_num=int(self.cfg.lidar.airy_n_elevation),
            horizontal_fov_deg_min=-180.0,
            horizontal_fov_deg_max=180.0,
            vertical_fov_deg_min=float(self.cfg.lidar.airy_vertical_fov_deg[0]),
            vertical_fov_deg_max=float(self.cfg.lidar.airy_vertical_fov_deg[1]),
            return_pointcloud=True,
            pointcloud_in_world_frame=False,
            randomize_placement=False,
            enable_sensor_noise=bool(self.cfg.lidar.enable_sensor_noise),
            pixel_std_dev_multiplier=float(
                self.cfg.lidar.pixel_std_dev_multiplier
            ),
            pixel_dropout_prob=float(self.cfg.lidar.pixel_dropout_prob),
            random_distance_noise=float(self.cfg.lidar.random_distance_noise),
            random_angle_noise=float(self.cfg.lidar.random_angle_noise),
        )

        lidar_env = {
            "device": self.device,
            "num_envs": self.num_envs,
            "num_sensors": 1,
            "sensor_pos_tensor": self.sensor_pos_tensor,
            "sensor_quat_tensor": self.sensor_quat_tensor,
            "mesh_ids": self.mesh_ids,
        }
        self.lidar_sensor = LidarSensor(
            lidar_env, None, lidar_cfg, num_sensors=1, device=self.device
        )

        offset_pos = list(self.cfg.lidar.offset_pos)
        self._sensor_translation = torch.tensor(
            offset_pos, dtype=torch.float32, device=self.device
        ).view(1, 3).repeat(self.num_envs, 1)

        rpy = list(self.cfg.lidar.sensor_offset_rpy)
        offset_q = quat_from_euler_xyz(
            torch.tensor(float(rpy[0]), device=self.device),
            torch.tensor(float(rpy[1]), device=self.device),
            torch.tensor(float(rpy[2]), device=self.device),
        )
        self._sensor_offset_quat = offset_q.view(1, 4).repeat(
            self.num_envs, 1
        )

        # Load the fixed Airy mapping table once.
        self.airy_mapping = (
            ea2c.EA2_MAPPING_TABLE_FILE.exists()
            and torch.load(ea2c.EA2_MAPPING_TABLE_FILE, map_location="cpu")
        )
        if not isinstance(self.airy_mapping, torch.Tensor):
            from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (
                build_airy_mapping_table,
            )

            self.airy_mapping = build_airy_mapping_table()

    def _update_lidar(self):
        """Advance the global 10 Hz clock and refresh the 450-dim range image."""
        self._lidar_timer += 1
        if self._lidar_timer % self._lidar_decimation != 0:
            return

        from isaacgym.torch_utils import quat_apply, quat_mul

        self.sensor_quat_tensor.copy_(
            quat_mul(self.base_quat, self._sensor_offset_quat)
        )
        self.sensor_pos_tensor.copy_(
            self.base_pos + quat_apply(self.base_quat, self._sensor_translation)
        )

        lidar_points, lidar_dist = self.lidar_sensor.update()
        points = lidar_points.view(self.num_envs, -1, 3)
        dists = lidar_dist.view(self.num_envs, -1)

        fresh = aggregate_range_image(
            points,
            dists,
            self.airy_mapping,
            max_range=float(self.cfg.lidar.effective_max_range),
            r_min=float(self.cfg.lidar.min_range),
        )
        # README 2.4 empty-frame contract: stale envs keep the all-max_range
        # empty frame only until this global scan, at which point they receive
        # the fresh aggregate computed from their new pose.
        refresh_range_image_from_scan(
            self.range_image,
            fresh,
            self.range_image_stale,
        )

        # Keep the latest noisy point cloud for debug-env visualization only.
        # Cloning all 5760 rays for every env would waste ~100 MB at 1024 envs.
        if self.viewer is not None and self._debug_env_ids:
            ids = torch.tensor(
                self._debug_env_ids, dtype=torch.long, device=self.device
            )
            self._debug_points = points[ids].clone()
            self._debug_dists = dists[ids].clone()
            self._debug_base_pos = self.base_pos[ids].clone()
            self._debug_base_quat = self.base_quat[ids].clone()

    # ------------------------------------------------------------------
    # Debug visualization (README 2.7 red/green point cloud)
    # ------------------------------------------------------------------

    def _draw_debug_vis(self) -> None:
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
        max_r = float(self.cfg.lidar.effective_max_range)
        r_min = float(self.cfg.lidar.min_range)
        stride = max(1, self._debug_point_stride)

        red_geom = gymutil.WireframeSphereGeometry(
            0.04, 4, 4, None, color=(1.0, 0.0, 0.0)
        )
        green_geom = gymutil.WireframeSphereGeometry(
            0.03, 4, 4, None, color=(0.0, 1.0, 0.0)
        )

        for k, eid in enumerate(self._debug_env_ids):
            if eid >= self.num_envs:
                continue
            dists = self._debug_dists[k]
            red, green = point_cloud_debug_masks(
                dists,
                self.airy_mapping,
                far_plane=far_plane,
                max_range=max_r,
                r_min=r_min,
            )

            pts_sensor = self._debug_points[k][::stride]
            red = red[::stride]
            green = green[::stride]
            n = pts_sensor.shape[0]

            # sensor frame -> body frame -> world frame
            offset_q = self._sensor_offset_quat[eid : eid + 1].expand(n, 4)
            pts_body = (
                quat_apply(offset_q, pts_sensor)
                + self._sensor_translation[eid : eid + 1]
            )
            base_q = self._debug_base_quat[k : k + 1].expand(n, 4)
            pts_world = self._debug_base_pos[k] + quat_apply(base_q, pts_body)
            pts_world = pts_world.detach().cpu().numpy()

            red_pts = pts_world[red.cpu().numpy()]
            green_pts = pts_world[green.cpu().numpy()]
            for pt in red_pts:
                pose = gymapi.Transform(
                    gymapi.Vec3(float(pt[0]), float(pt[1]), float(pt[2]))
                )
                gymutil.draw_lines(
                    red_geom, self.gym, self.viewer, self.envs[eid], pose
                )
            for pt in green_pts:
                pose = gymapi.Transform(
                    gymapi.Vec3(float(pt[0]), float(pt[1]), float(pt[2]))
                )
                gymutil.draw_lines(
                    green_geom, self.gym, self.viewer, self.envs[eid], pose
                )

            # Current policy envelope footprint (cyan) at z=0.02 m.
            hex_xy = self._compute_hex_world()[eid].numpy()  # (6, 2)
            z = 0.02
            line_verts = []
            line_colors = []
            for i in range(6):
                j = (i + 1) % 6
                line_verts.extend(
                    [
                        float(hex_xy[i, 0]),
                        float(hex_xy[i, 1]),
                        z,
                        float(hex_xy[j, 0]),
                        float(hex_xy[j, 1]),
                        z,
                    ]
                )
                line_colors.extend([0.0, 0.85, 1.0, 0.0, 0.85, 1.0])
            if line_verts:
                self.gym.add_lines(
                    self.viewer,
                    self.envs[eid],
                    6,
                    np.asarray(line_verts, dtype=np.float32),
                    np.asarray(line_colors, dtype=np.float32),
                )

    # ------------------------------------------------------------------
    # Path / reset
    # ------------------------------------------------------------------

    def _sample_free_start_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a start/goal pair that can plausibly yield a long path."""
        inflated = self.inflated
        free = np.argwhere(inflated == 0)
        if free.shape[0] == 0:
            raise RuntimeError("map has no free cells")
        start_idx = free[self._rng.integers(0, free.shape[0])]
        start_xy = (
            ea2c.EA2_WORLD_MIN_XY
            + (start_idx[1] + 0.5) * ea2c.EA2_RESOLUTION_M,
            ea2c.EA2_WORLD_MIN_XY
            + (start_idx[0] + 0.5) * ea2c.EA2_RESOLUTION_M,
        )

        # Goal must be clear of raw obstacles by cfg.path.goal_min_obstacle_dist.
        goal_clearance = float(self.cfg.path.goal_min_obstacle_dist)
        for _ in range(200):
            goal_idx = free[self._rng.integers(0, free.shape[0])]
            goal_xy = (
                ea2c.EA2_WORLD_MIN_XY
                + (goal_idx[1] + 0.5) * ea2c.EA2_RESOLUTION_M,
                ea2c.EA2_WORLD_MIN_XY
                + (goal_idx[0] + 0.5) * ea2c.EA2_RESOLUTION_M,
            )
            if self._min_obstacle_distance_world(goal_xy) >= goal_clearance:
                return np.asarray(start_xy, dtype=np.float64), np.asarray(
                    goal_xy, dtype=np.float64
                )
        raise RuntimeError("could not sample a valid goal in 200 attempts")

    def _min_obstacle_distance_world(self, xy) -> float:
        cells = np.argwhere(self.occupancy > 0)
        if cells.size == 0:
            return float("inf")
        xs = ea2c.EA2_WORLD_MIN_XY + (cells[:, 1].astype(np.float64) + 0.5) * ea2c.EA2_RESOLUTION_M
        ys = ea2c.EA2_WORLD_MIN_XY + (cells[:, 0].astype(np.float64) + 0.5) * ea2c.EA2_RESOLUTION_M
        return float(np.min(np.hypot(xs - float(xy[0]), ys - float(xy[1]))))

    def _sample_new_path(self) -> PathData:
        """Sample a feasible start/goal and plan one noisy A* path."""
        path_cfg = PathCfg(
            speed_range=tuple(self.cfg.path.speed_range),
            resample_time_s=float(self.cfg.path.resample_time_s),
            delta_target_deg_range=tuple(self.cfg.path.delta_target_deg_range),
            omega_max=float(self.cfg.path.omega_max),
            k_p=float(self.cfg.path.k_p),
            min_turn_radius=float(self.cfg.path.min_turn_radius),
            resample_dist=float(self.cfg.path.resample_dist),
            goal_min_obstacle_dist=float(self.cfg.path.goal_min_obstacle_dist),
            min_path_len=float(self.cfg.path.min_path_len),
            noise_amp_range=tuple(self.cfg.path.noise_amp_range),
            noise_fc_hz=float(self.cfg.path.noise_fc_hz),
            noise_retries=int(self.cfg.path.noise_retries),
        )
        last_err: Optional[Exception] = None
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
        raise RuntimeError(
            f"failed to plan a path after 40 attempts: {last_err}"
        )

    def _reset_one_env(self, env_id: int):
        path = self._sample_new_path()
        self.paths[env_id] = path

        start_xy = path.points[0]
        start_yaw = float(path.yaws[0])
        self.s[env_id] = 0.0
        self.base_pos[env_id, 0] = float(start_xy[0])
        self.base_pos[env_id, 1] = float(start_xy[1])

        # Body quaternion from yaw (z-up).
        half = start_yaw * 0.5
        self.base_quat[env_id, 0] = 0.0
        self.base_quat[env_id, 1] = 0.0
        self.base_quat[env_id, 2] = math.sin(half)
        self.base_quat[env_id, 3] = math.cos(half)

        self.heading[env_id] = start_yaw
        self.tangent[env_id] = start_yaw
        self.tangent_rate[env_id] = 0.0
        self.delta_actual[env_id] = 0.0
        self.omega[env_id] = 0.0
        self.v[env_id] = float(
            self._rng.uniform(*self.cfg.path.speed_range)
        )
        deg = float(
            self._rng.uniform(*self.cfg.path.delta_target_deg_range)
        )
        self.delta_target[env_id] = math.radians(deg)

        self.height_target[env_id] = float(
            self._rng.uniform(self.cfg.height.min_m, self.cfg.height.max_m)
        )
        self.height_filter[env_id] = float(self.height_target[env_id])
        self.base_pos[env_id, 2] = float(self.height_target[env_id])
        self.height_wobble_state[env_id] = 0.0
        self.pos_sway_state[env_id] = 0.0
        self.heading_sway_state[env_id] = 0.0
        self.pos_sway_amp[env_id] = float(
            self._rng.uniform(*self.cfg.sway.pos_amp_range)
        )
        self.heading_sway_amp[env_id] = float(
            self._rng.uniform(*self.cfg.sway.heading_amp_range)
        )
        self.height_wobble_amp[env_id] = float(
            self._rng.uniform(*self.cfg.height.wobble_amp_range)
        )

        self.episode_length_buf[env_id] = 0
        # Keep the done flag set for the caller (PPO recurrent reset).  The
        # buffer is overwritten on the next step before a new decision, so
        # leaving it 1 does not carry a spurious reset into the next episode.
        self.reset_buf[env_id] = 1
        # Do NOT clear time_out_buf here: step() returns it as infos and PPO
        # reads time_outs immediately after this reset call.
        self.actions_mapped[env_id] = map_actions_to_params(
            torch.zeros_like(self.actions_mapped[env_id]),
            self._envelope_low.to(self.device),
            self._envelope_high.to(self.device),
        )
        # last_actions stores the mapped params used by action_rate; initialise
        # it to the reset mapped midpoint so the first step has no cross-episode
        # action-rate penalty.
        self.last_actions[env_id] = self.actions_mapped[env_id].clone()
        self.range_image_stale[env_id] = True
        self.range_image[env_id] = float(self.cfg.lidar.effective_max_range)

    def reset_idx(self, env_ids):
        """Reset selected envs and log episode metrics."""
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(self.device)
        ids = env_ids.tolist()

        # Log before resetting the buffers.
        if self.episode_sums["potential"][env_ids].sum() > 0.0 or True:
            ep = {}
            for name in self.reward_scales.keys():
                ep[f"rew_{name}"] = (
                    self.episode_sums[name][env_ids].mean()
                    / max(self.max_episode_length, 1)
                )
            self.extras["episode"] = ep

        for env_id in ids:
            self._reset_one_env(env_id)

        # Fresh ego-motion for newly reset episodes (old episode values must
        # not leak into the observation returned after reset).
        for i in ids:
            vx, vy, omega_out = ego_motion(
                self.v[i : i + 1],
                self.heading[i : i + 1],
                self.tangent[i : i + 1],
                self.omega[i : i + 1],
            )
            self.ego_motion[i, 0] = vx
            self.ego_motion[i, 1] = vy
            self.ego_motion[i, 2] = omega_out

        for name in self.reward_scales.keys():
            self.episode_sums[name][env_ids] = 0.0

        # Assign after the resets so the returned tensor is the live timeout
        # buffer; reset_one_env intentionally does not clear it.
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    # ------------------------------------------------------------------
    # Kinematics / observation / reward
    # ------------------------------------------------------------------

    def _step_kinematics(self):
        """Advance each env along its reference path with sway/height."""
        n = self.num_envs
        dt = self.dt
        rng = torch.Generator(device=self.device)
        rng.manual_seed(int(self.common_step_counter) * 7919 + 1)

        for i in range(n):
            path = self.paths[i]
            if path is None:
                continue

            # Advance arc length.
            self.s[i] = self.s[i] + self.v[i] * dt
            if self.s[i] >= float(path.arc[-1]):
                self.s[i] = float(path.arc[-1])
                self.reset_buf[i] = 1

            xy, tangent, tangent_rate = interpolate_path(
                path, float(self.s[i])
            )
            self.tangent[i] = tangent
            self.tangent_rate[i] = tangent_rate * float(self.v[i])

            # Heading controller.
            heading, omega, delta_actual = heading_update(
                self.heading[i : i + 1],
                self.tangent[i : i + 1],
                self.tangent_rate[i : i + 1],
                self.delta_target[i : i + 1],
                dt,
                float(self.cfg.path.k_p),
                float(self.cfg.path.omega_max),
            )
            self.heading[i] = heading
            self.omega[i] = omega
            self.delta_actual[i] = delta_actual

            # Heading sway (applied after controller, then delta recomputed).
            noise = torch.randn((1,), generator=rng, device=self.device)
            self.heading_sway_state[i : i + 1], heading_offset = sway_update(
                self.heading_sway_state[i : i + 1],
                float(self.heading_sway_amp[i]),
                dt,
                float(self.cfg.sway.fc_hz),
                noise,
            )
            self.heading[i] = float(
                wrap_to_pi(self.heading[i] + heading_offset[0])
            )
            self.delta_actual[i] = float(
                wrap_to_pi(self.heading[i] - self.tangent[i])
            )

            # Lateral position sway with rejection to the last legal pose.
            pos_noise = torch.randn((1,), generator=rng, device=self.device)
            self.pos_sway_state[i : i + 1], pos_offset = sway_update(
                self.pos_sway_state[i : i + 1],
                float(self.pos_sway_amp[i]),
                dt,
                float(self.cfg.sway.fc_hz),
                pos_noise,
            )
            normal = np.array(
                [-math.sin(tangent), math.cos(tangent)], dtype=np.float64
            )
            cand = np.asarray(xy, dtype=np.float64) + normal * float(pos_offset[0])
            if sway_position_acceptable(
                (float(self.base_pos[i, 0]), float(self.base_pos[i, 1])),
                cand,
                self.inflated,
            ):
                self.base_pos[i, 0] = float(cand[0])
                self.base_pos[i, 1] = float(cand[1])
            # If rejected, keep the previous legal position (no write).

            # Height filter/wobble.
            h_noise = torch.randn((1,), generator=rng, device=self.device)
            self.height_filter[i : i + 1], self.height_wobble_state[i : i + 1], h = (
                height_step(
                    self.height_filter[i : i + 1],
                    self.height_target[i : i + 1],
                    self.height_wobble_state[i : i + 1],
                    h_noise,
                    float(self.cfg.height.min_m),
                    float(self.cfg.height.max_m),
                    dt,
                    float(self.cfg.height.tau_s),
                    float(self.cfg.height.wobble_fc_hz),
                    float(self.height_wobble_amp[i]),
                )
            )
            self.base_pos[i, 2] = float(h[0])

            # Resample speed / delta_target / height target every 4 s.
            resample_period = float(self.cfg.path.resample_time_s)
            if self.episode_length_buf[i] > 0 and (
                int(self.episode_length_buf[i]) % max(1, int(resample_period / dt))
            ) == 0:
                self.v[i] = float(self._rng.uniform(*self.cfg.path.speed_range))
                deg = float(
                    self._rng.uniform(*self.cfg.path.delta_target_deg_range)
                )
                self.delta_target[i] = math.radians(deg)
                self.height_target[i] = float(
                    self._rng.uniform(
                        self.cfg.height.min_m, self.cfg.height.max_m
                    )
                )

            # Ego motion (body frame).
            vx, vy, omega_out = ego_motion(
                self.v[i : i + 1],
                self.heading[i : i + 1],
                self.tangent[i : i + 1],
                self.omega[i : i + 1],
            )
            self.ego_motion[i, 0] = vx
            self.ego_motion[i, 1] = vy
            self.ego_motion[i, 2] = omega_out

    def _compute_hex_world(self) -> torch.Tensor:
        """Return world-frame offset hexagon vertices for all envs (CPU)."""
        params = self.actions_mapped.to("cpu")
        low = self._envelope_low.to("cpu")
        high = self._envelope_high.to("cpu")
        # Map raw buffer into actual params? actions_mapped already mapped.
        body_hex = compute_hex_vertices(
            params[:, 0],
            params[:, 1],
            params[:, 2],
            params[:, 3],
            params[:, 4],
        )  # (E,6,2)
        heading = self.heading.to("cpu")
        base_xy = self.base_pos[:, :2].to("cpu")
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        rot = torch.stack(
            [torch.stack([cos_h, -sin_h], dim=-1),
             torch.stack([sin_h, cos_h], dim=-1)],
            dim=-2,
        )  # (E,2,2)
        world = torch.einsum("eij,evj->evi", rot, body_hex) + base_xy.unsqueeze(1)
        return world

    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        low = self._envelope_low.to(self.device)
        high = self._envelope_high.to(self.device)

        potential = potential_reward(self.actions_mapped, low, high)
        coll_ratio = collision_ratio(
            self._compute_hex_world(),
            self.occupancy,
            margin=float(self.cfg.envelope.margin),
        ).to(self.device)
        act_rate = action_rate_term(self.actions_mapped, self.last_actions)

        terms = {
            "potential": potential,
            "collision": coll_ratio,
            "action_rate": act_rate,
        }
        for name, scale in self.reward_scales.items():
            rew = terms[name] * scale
            self.rew_buf += rew
            self.episode_sums[name] += rew

    def _compute_observations(self):
        self.obs_buf[:] = assemble_observation(
            self.range_image,
            self.ego_motion,
            max_range=float(self.cfg.lidar.effective_max_range),
        )

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def compute_observations(self):
        """Alias for :meth:`_compute_observations` (LeggedRobot-style API)."""
        self._compute_observations()

    def compute_reward(self):
        """Alias for :meth:`_compute_rewards` (LeggedRobot-style API)."""
        self._compute_rewards()

    def step(self, actions):
        """Old legged_gym 5-tuple step interface (README 2.9)."""
        clip_actions = float(self.cfg.normalization.clip_actions)
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(
            self.device
        )
        self.actions_mapped = map_actions_to_params(
            self.actions,
            self._envelope_low.to(self.device),
            self._envelope_high.to(self.device),
        )

        self.render()
        self.common_step_counter += 1
        self.episode_length_buf += 1

        self._step_kinematics()
        self._update_lidar()

        # Clear timeout before deciding dones (README 2.9).
        self.time_out_buf[:] = False
        reached_end = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        for i in range(self.num_envs):
            path = self.paths[i]
            if path is not None and float(self.s[i]) >= float(path.arc[-1]):
                reached_end[i] = True
        self.reset_buf = reached_end.to(torch.long)
        timeout = self.episode_length_buf >= self.max_episode_length
        self.time_out_buf = timeout
        self.reset_buf |= timeout.to(torch.long)

        self._compute_rewards()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self._compute_observations()

        # reset_idx has already replaced actions_mapped for reset envs with the
        # reset midpoint, so this update makes last_actions match the new
        # episode's first mapped params and avoids a cross-episode penalty.
        self.last_actions[:] = self.actions_mapped[:]

        clip_obs = float(self.cfg.normalization.clip_observations)
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )

        self._draw_debug_vis()

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )
