"""Frozen public contracts for ``envelope_adaptive_2``.

All implementation agents MUST keep their modules compatible with the
dataclasses, constants and function signatures documented in this file.
No implementation agent may edit this file; it is owned by the integration
owner (v4-pro) only.

The actual implementations live in their own modules:
    - ``utils/LidarSensor/lidar_sensor.py``  -> ``apply_noise`` (shared util patch)
    - ``airy_mount.py``                      -> mapping table + mount self-test
    - ``envelope_geometry.py``               -> hexagon / offset / grid collision
    - ``map_generator.py``                   -> primitives -> occupancy -> warp mesh
    - ``path_planner.py``                    -> A* / smoothing / noise / heading
    - ``range_image.py``                     -> 86400 full rays -> 187 fixed channels
    - ``el_4090_ea2_env.py``                 -> BaseTask integration
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths / constants (single source of truth)
# ---------------------------------------------------------------------------
EA2_DIR = Path(__file__).resolve().parent
EA2_SELECTED_CHANNELS_FILE = EA2_DIR / "selected_airy_channels.pt"

EA2_MAP_SIZE_M = 74.0    # 4 x 4 tiles of 16m x 16m + 5m border on each side
EA2_RESOLUTION_M = 0.1
EA2_GRID_SHAPE = (740, 740)  # (rows=iy, cols=ix)
EA2_WORLD_MIN_XY = -37.0
EA2_WORLD_MAX_XY = 37.0
EA2_GROUND_MARGIN_M = 2.0
EA2_BASE_HEIGHT_M = 0.52

# Full Airy pattern used for offline channel selection (0.4 deg horizontal).
EA2_AIRY_N_AZIMUTH_FULL = 900
EA2_AIRY_N_ELEVATION = 96
EA2_AIRY_HORIZONTAL_RES_DEG = 0.4
EA2_FULL_N_RAYS = EA2_AIRY_N_AZIMUTH_FULL * EA2_AIRY_N_ELEVATION  # 86400
EA2_RAY_INDEX = "i = az * 96 + el"  # LidarSensor C-order flatten

# Fixed 187-channel ground grid (11 rows along x, 17 cols along y).
EA2_GRID_ROWS = 11
EA2_GRID_COLS = 17
EA2_RANGE_DIM = EA2_GRID_ROWS * EA2_GRID_COLS  # 187
EA2_REGION_X_MIN = 0.65
EA2_REGION_X_MAX = 3.65
EA2_REGION_Y_MIN = -1.0
EA2_REGION_Y_MAX = 1.0

# Normalization divisor = max slant distance among selected channels, rounded
# up to 0.1 m.  Computed by airy_mount during channel selection.
EA2_RANGE_MAX_M = 3.2
EA2_LIDAR_FAR_PLANE_M = 60.0

# 4x4 tile terrain layout (pd_gru pillar-field random cuboids per tile)
EA2_N_TILES = 4
EA2_TILE_EMPTY = 0
EA2_TILE_WALL = 1
EA2_TILE_PILLAR = 2
EA2_TILE_CORRIDOR = 3
EA2_TILE_SIDE_WALLS = 4
EA2_TILE_U_SHAPE = 5
EA2_TILE_TYPE_CODES = (
    EA2_TILE_EMPTY,
    EA2_TILE_WALL,
    EA2_TILE_PILLAR,
    EA2_TILE_CORRIDOR,
    EA2_TILE_SIDE_WALLS,
    EA2_TILE_U_SHAPE,
)

# Sensor mount (body frame; current EA2 placement)
EA2_SENSOR_OFFSET_POS = (0.7, 0.0, -0.05)
EA2_SENSOR_OFFSET_RPY = (0.0, math.pi / 2.0 + 0.1, 0.0)

# Envelope (must stay identical to spider_envelop config; do not duplicate)
ENVELOPE_SPEC_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "envs" / "el_4090" / "spider_envelop" / "el4090_spider_config.py"
)

# ---------------------------------------------------------------------------
# Dataclasses shared between modules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RectPrimitive:
    """Wall / U-shape segment footprint. (x, y) world center, size=full extents."""

    center: Tuple[float, float]
    size: Tuple[float, float]  # (length_x, length_y)
    yaw: float = 0.0
    height: float = 1.5


@dataclass(frozen=True)
class PillarPrimitive:
    """Square/circular pillar footprint. ``radius`` is half-side or radius."""

    center: Tuple[float, float]
    radius: float
    height: float
    square: bool = False
    segments: int = 16


@dataclass
class MapData:
    """Output of ``map_generator.generate_map``.

    Arrays use world coordinates x/y in [-37, 37]; grid indexing follows
    ``ix = floor((x + 37) / 0.1)``.  ``inflated`` is the 0.35 m (4-cell)
    safety grid for A*/path checks and is computed once per fixed map.
    """

    occupancy: np.ndarray           # (740, 740) uint8, 1 = occupied
    inflated: np.ndarray            # (740, 740) uint8, 1 = blocked (planning)
    vertices: np.ndarray            # (V, 3) float32, watertight ground+obstacles
    triangles: np.ndarray           # (T, 3) int32, CCW/outward-consistent winding
    tile_types: Optional[np.ndarray] = None  # (5, 5) uint8 tile type codes
    rects: Tuple[RectPrimitive, ...] = ()
    pillars: Tuple[PillarPrimitive, ...] = ()
    acceptance: Dict[str, float] = field(default_factory=dict)


@dataclass
class PathData:
    """Output of ``path_planner.plan_path`` (one env, one episode)."""

    points: np.ndarray              # (P, 2) world x/y
    yaws: np.ndarray                # (P,) tangent yaw, radians
    arc: np.ndarray                 # (P,) cumulative arc length, starts at 0


@dataclass(frozen=True)
class PillarFieldCfg:
    """pd_gru_lidar pillar-field parameters (per 16m x 16m terrain tile)."""

    count_min: int = 0
    count_max: int = 12
    size_x_min: float = 0.5
    size_x_max: float = 4.0
    size_y_min: float = 0.5
    size_y_max: float = 4.0
    height_min: float = 1.0
    height_max: float = 2.0
    min_separation: float = 2.2
    center_clear_radius: float = 3.0
    spawn_radius: float = 7.5
    allow_height_variation: bool = True


@dataclass(frozen=True)
class MapGenCfg:
    size_m: float = EA2_MAP_SIZE_M
    resolution_m: float = EA2_RESOLUTION_M
    grid_shape: Tuple[int, int] = EA2_GRID_SHAPE
    boundary_occupied: bool = True   # planning border in *inflated* grid only
    ground_margin_m: float = EA2_GROUND_MARGIN_M
    inflation_m: float = 0.35
    inflation_cells: int = 4
    n_tiles: int = EA2_N_TILES       # n_tiles x n_tiles pillar-field plots
    tile_size_m: float = 16.0        # pd_gru terrain_length / terrain_width
    border_size_m: float = 5.0       # pd_gru border_size
    max_gen_attempts: int = 20
    n_validation_paths: int = 12
    min_solved_ratio: float = 0.8
    path_near_obstacle_ratio: float = 0.3
    near_obstacle_range: Tuple[float, float] = (0.7, 1.5)
    require_constraint_primitive: bool = False  # pillar field has no corridor type
    min_free_component_ratio: float = 0.95  # inflated free-space connectivity


@dataclass(frozen=True)
class PathCfg:
    speed_range: Tuple[float, float] = (0.5, 1.5)
    resample_time_s: float = 4.0
    delta_target_deg_range: Tuple[float, float] = (-20.0, 20.0)
    omega_max: float = 1.5
    k_p: float = 5.0
    min_turn_radius: float = 1.0
    resample_dist: float = 0.2
    goal_min_obstacle_dist: float = 0.5
    min_path_len: float = 3.0
    noise_amp_range: Tuple[float, float] = (0.15, 0.25)
    noise_fc_hz: float = 1.0
    noise_retries: int = 8


@dataclass(frozen=True)
class SwayCfg:
    pos_amp_range: Tuple[float, float] = (0.02, 0.05)
    heading_amp_range: Tuple[float, float] = (0.05, 0.1)
    fc_hz: float = 1.0


@dataclass(frozen=True)
class HeightCfg:
    min_m: float = 0.53
    max_m: float = 0.64
    resample_time_s: float = 4.0
    tau_s: float = 0.8
    wobble_amp_range: Tuple[float, float] = (0.01, 0.02)
    wobble_fc_hz: float = 1.0


@dataclass(frozen=True)
class LidarNoiseCfg:
    enable: bool = True
    pixel_std_dev_multiplier: float = 0.02
    pixel_dropout_prob: float = 0.02
    random_distance_noise: float = 0.0
    random_angle_noise: float = 0.0


# ---------------------------------------------------------------------------
# Function contracts (implementation modules MUST provide these signatures)
# ---------------------------------------------------------------------------
#
# utils/LidarSensor/lidar_sensor.py
#   def apply_noise(self, pixels: torch.Tensor, dists: torch.Tensor) \
#           -> Tuple[torch.Tensor, torch.Tensor]
#       - pixels (E,1,N,1,3), dists (E,1,N,1); gated by cfg.enable_sensor_noise
#       - multiplicative Gaussian on range, dropout -> far_plane, all rays
#       - reference implementation in README section 2.2.8
#
# airy_mount.py
#   def generate_full_airy_directions() -> torch.Tensor     # (86400, 3)
#   def body_frame_ray_directions() -> torch.Tensor         # (86400, 3)
#   def select_ground_grid_channels() -> Dict[str, object]
#       - selects 187 unique channels for the 11x17 ground grid
#   def load_selected_channels() -> Dict[str, object]
#   def self_check_selected_channels(data) -> Dict[str, object]
#
# envelope_geometry.py
#   def compute_hex_vertices(front_width, middle_width, back_width,
#                            forward_limit, backward_limit) -> torch.Tensor
#       # (..., 6, 2), vertex order B,D,F,E,C,A (legacy-compatible)
#   def offset_hexagon(vertices, margin) -> torch.Tensor    # half-plane exact offset
#   def point_in_hex(pts_xy, vertices) -> torch.Tensor      # bool mask (..., N)
#   def collision_cell_ratio(hex_vertices_world_xy, occupancy,
#                            world_to_grid_fn) -> torch.Tensor
#       # covered occupied cells / covered cells, eps-protected
#   def envelope_params_to_condition(params5: torch.Tensor,
#                                    spec) -> torch.Tensor # (..., 8)
#       # must reuse apply_env_morphology_priors (do not re-implement priors)
#
# map_generator.py
#   def generate_map(cfg: MapGenCfg, pillar_cfg: PillarFieldCfg,
#                    seed: int) -> MapData
#
# path_planner.py
#   def plan_path(occupancy: np.ndarray, inflated: np.ndarray,
#                 start_xy, goal_xy, cfg: PathCfg,
#                 rng: np.random.Generator) -> PathData
#   def heading_update(heading, tangent, tangent_rate, delta_target,
#                      v, dt, k_p, omega_max) -> Tuple[heading, omega, delta_actual]
#   def ego_motion(v, heading, tangent, omega) -> Tuple[vx, vy, omega]
#
# range_image.py
#   def build_selected_range_image(dists, max_range) -> torch.Tensor
#       # (E,187) -> (E,187), no-hit/out-of-range = max_range
#   def extract_selected_range_image(full_dists, selected_indices,
#                                    max_range) -> torch.Tensor
#       # (E,86400) + (187,) -> (E,187)
#   def range_image_observation(range_image, max_range) -> torch.Tensor
#
# el_4090_ea2_env.py
#   class EL_4090_EA2(BaseTask)
#       - old legged_gym interface: step returns 5-tuple
#       - defines self.dt, self.max_episode_length, time_outs in infos
#       - all envs share world origin (env_origins=0), warp mesh authoritative
