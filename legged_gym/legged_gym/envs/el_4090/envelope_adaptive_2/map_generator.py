"""Deterministic seeded map generator for envelope_adaptive_2 (M1).

Layout contract (user-confirmed revision):
  * one global fixed 12m x 12m map split into a ``n_tiles x n_tiles`` grid
    (default 5 x 5, 2.4m tiles);
  * each tile contains exactly ONE terrain type (empty / wall / pillar /
    corridor / side-walls / U-shape), so primitives never mix or spill into
    neighbouring tiles;
  * all rect primitives are axis-aligned (yaw 0 or pi/2) -- no slanted walls;
  * there are NO physical boundary walls.  The planning (inflated) grid border
    is still marked blocked so A* keeps the robot inside the map, but nothing
    is rasterized or meshed at the border.

The occupancy grid remains the authoritative geometry; the Warp mesh is built
from the same axis-aligned box / n-gon prism primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ._contracts import (
    EA2_TILE_CORRIDOR,
    EA2_TILE_EMPTY,
    EA2_TILE_PILLAR,
    EA2_TILE_SIDE_WALLS,
    EA2_TILE_U_SHAPE,
    EA2_TILE_WALL,
    MapData,
    MapGenCfg,
    PillarPrimitive,
    RectPrimitive,
)

_HEIGHT_RANGE = (1.5, 2.0)
_PILLAR_SEGMENTS = 16
_GROUND_THICKNESS_M = 0.02
_NEAR_OBSTACLE_SAMPLE_CAP = 2000

# Tile type frequencies for the 5x5 layout (sums to 25).
_TILE_COUNTS = {
    EA2_TILE_EMPTY: 6,
    EA2_TILE_WALL: 5,
    EA2_TILE_PILLAR: 4,
    EA2_TILE_CORRIDOR: 4,
    EA2_TILE_SIDE_WALLS: 3,
    EA2_TILE_U_SHAPE: 3,
}


@dataclass(frozen=True)
class _Mesh:
    """Small internal mesh container used while assembling the final arrays."""

    vertices: np.ndarray
    triangles: np.ndarray


def _world_to_grid(x: float, y: float, cfg: MapGenCfg) -> Tuple[int, int]:
    """Convert a world coordinate to grid indices using the contract formula."""
    half = cfg.size_m / 2.0
    ix = int(np.floor((x + half) / cfg.resolution_m))
    iy = int(np.floor((y + half) / cfg.resolution_m))
    return ix, iy


def _grid_centers(cfg: MapGenCfg) -> Tuple[np.ndarray, np.ndarray]:
    """Return world coordinates of every grid-cell center."""
    world_min = -cfg.size_m / 2.0
    xs = world_min + (np.arange(cfg.grid_shape[1]) + 0.5) * cfg.resolution_m
    ys = world_min + (np.arange(cfg.grid_shape[0]) + 0.5) * cfg.resolution_m
    return xs, ys


def _circle_polygon(radius: float, segments: int) -> np.ndarray:
    """Return the CCW inscribed n-gon used for circular pillar footprints."""
    n = max(3, int(segments))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius


# ---------------------------------------------------------------------------
# 5x5 tile layout helpers
# ---------------------------------------------------------------------------


def _make_tile_layout(rng: np.random.Generator) -> np.ndarray:
    """Build a shuffled ``(n_tiles, n_tiles)`` tile-type grid."""
    tiles: List[int] = []
    for code, count in _TILE_COUNTS.items():
        tiles.extend([int(code)] * int(count))
    rng.shuffle(tiles)
    n = int(round(math.sqrt(len(tiles))))
    if n * n != len(tiles):
        raise ValueError(f"tile counts must form a square grid, got {len(tiles)}")
    return np.asarray(tiles, dtype=np.uint8).reshape(n, n)


def _tile_center(cfg: MapGenCfg, row: int, col: int) -> Tuple[float, float]:
    """World center of tile ``(row, col)``; row 0 is the most negative y."""
    tile = cfg.size_m / float(cfg.n_tiles)
    x = -cfg.size_m / 2.0 + (float(col) + 0.5) * tile
    y = -cfg.size_m / 2.0 + (float(row) + 0.5) * tile
    return x, y


def _axis_dirs(yaw: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return ``(direction, normal)`` for an axis-aligned yaw (0 or pi/2)."""
    direction = (math.cos(yaw), math.sin(yaw))
    normal = (-math.sin(yaw), math.cos(yaw))
    return direction, normal


# ---------------------------------------------------------------------------
# One primitive set per tile type (all axis-aligned)
# ---------------------------------------------------------------------------


def _tile_wall(
    rng: np.random.Generator, center: Tuple[float, float], height: float
) -> List[RectPrimitive]:
    length = float(rng.uniform(1.2, 1.8))
    thickness = float(rng.uniform(0.2, 0.4))
    yaw = 0.0 if rng.integers(0, 2) == 0 else math.pi / 2.0
    return [
        RectPrimitive(
            center=center, size=(length, thickness), yaw=yaw, height=height
        )
    ]


def _tile_pillars(
    rng: np.random.Generator,
    center: Tuple[float, float],
    height: float,
    square: bool,
) -> List[PillarPrimitive]:
    n = int(rng.integers(1, 3))  # 1 or 2 pillars
    horizontal = bool(rng.integers(0, 2) == 0)
    offsets = [(-0.25, 0.0), (0.25, 0.0)] if horizontal else [
        (0.0, -0.25), (0.0, 0.25)
    ]
    pillars: List[PillarPrimitive] = []
    for i in range(n):
        radius = float(rng.uniform(0.15, 0.30))
        pillars.append(
            PillarPrimitive(
                center=(center[0] + offsets[i][0], center[1] + offsets[i][1]),
                radius=radius,
                height=height,
                square=square,
                segments=_PILLAR_SEGMENTS,
            )
        )
    return pillars


def _tile_corridor(
    rng: np.random.Generator, center: Tuple[float, float], height: float
) -> List[RectPrimitive]:
    """Two parallel axis-aligned walls forming a narrow channel."""
    gap = float(rng.uniform(1.0, 1.4))
    thickness = float(rng.uniform(0.2, 0.3))
    length = 1.8
    yaw = 0.0 if rng.integers(0, 2) == 0 else math.pi / 2.0
    _, normal = _axis_dirs(yaw)
    offset = gap / 2.0 + thickness / 2.0
    return [
        RectPrimitive(
            center=(
                center[0] + normal[0] * offset,
                center[1] + normal[1] * offset,
            ),
            size=(length, thickness),
            yaw=yaw,
            height=height,
        ),
        RectPrimitive(
            center=(
                center[0] - normal[0] * offset,
                center[1] - normal[1] * offset,
            ),
            size=(length, thickness),
            yaw=yaw,
            height=height,
        ),
    ]


def _tile_side_walls(
    rng: np.random.Generator, center: Tuple[float, float], height: float
) -> List[RectPrimitive]:
    """2-3 short axis-aligned wall segments on one side of the tile."""
    n = int(rng.integers(2, 4))
    yaw = 0.0 if rng.integers(0, 2) == 0 else math.pi / 2.0
    direction, normal = _axis_dirs(yaw)
    offsets = (-0.4, 0.4) if n == 2 else (-0.5, 0.0, 0.5)
    lateral = 0.6
    seg_len = 0.8
    rects: List[RectPrimitive] = []
    for off in offsets[:n]:
        thickness = float(rng.uniform(0.2, 0.25))
        rects.append(
            RectPrimitive(
                center=(
                    center[0] + direction[0] * off + normal[0] * lateral,
                    center[1] + direction[1] * off + normal[1] * lateral,
                ),
                size=(seg_len, thickness),
                yaw=yaw,
                height=height,
            )
        )
    return rects


def _tile_u_shape(
    rng: np.random.Generator, center: Tuple[float, float], height: float
) -> List[RectPrimitive]:
    """Axis-aligned U: back wall plus two parallel side walls."""
    opening = float(rng.uniform(0.8, 1.0))
    depth = 1.2
    thickness = float(rng.uniform(0.2, 0.25))
    yaw = 0.0 if rng.integers(0, 2) == 0 else math.pi / 2.0
    direction, normal = _axis_dirs(yaw)

    back = RectPrimitive(
        center=(
            center[0] - direction[0] * depth / 2.0,
            center[1] - direction[1] * depth / 2.0,
        ),
        size=(opening + 2.0 * thickness, thickness),
        yaw=yaw,
        height=height,
    )
    side_offset = opening / 2.0 + thickness / 2.0
    side1 = RectPrimitive(
        center=(
            center[0] + normal[0] * side_offset,
            center[1] + normal[1] * side_offset,
        ),
        size=(depth, thickness),
        yaw=yaw,
        height=height,
    )
    side2 = RectPrimitive(
        center=(
            center[0] - normal[0] * side_offset,
            center[1] - normal[1] * side_offset,
        ),
        size=(depth, thickness),
        yaw=yaw,
        height=height,
    )
    return [back, side1, side2]


def _generate_tile_primitive_set(
    cfg: MapGenCfg,
    rng: np.random.Generator,
    tile_types: np.ndarray,
    height: float,
) -> Tuple[List[RectPrimitive], List[PillarPrimitive], dict]:
    """Generate axis-aligned primitives, one terrain type per tile."""
    rects: List[RectPrimitive] = []
    pillars: List[PillarPrimitive] = []
    counts = {code: 0 for code in _TILE_COUNTS}
    pillar_tile_index = 0

    for row in range(tile_types.shape[0]):
        for col in range(tile_types.shape[1]):
            code = int(tile_types[row, col])
            center = _tile_center(cfg, row, col)
            counts[code] += 1
            if code == EA2_TILE_EMPTY:
                continue
            if code == EA2_TILE_WALL:
                rects.extend(_tile_wall(rng, center, height))
            elif code == EA2_TILE_PILLAR:
                # Alternate square/circular pillar tiles so both mesh code
                # paths are always exercised.
                square = bool(pillar_tile_index % 2 == 0)
                pillars.extend(_tile_pillars(rng, center, height, square))
                pillar_tile_index += 1
            elif code == EA2_TILE_CORRIDOR:
                rects.extend(_tile_corridor(rng, center, height))
            elif code == EA2_TILE_SIDE_WALLS:
                rects.extend(_tile_side_walls(rng, center, height))
            elif code == EA2_TILE_U_SHAPE:
                rects.extend(_tile_u_shape(rng, center, height))
            else:  # pragma: no cover - layout builder only emits known codes
                raise ValueError(f"unknown tile type code {code}")
    return rects, pillars, counts


# ---------------------------------------------------------------------------
# Rasterization / inflation (cell-center-in-footprint semantics)
# ---------------------------------------------------------------------------


def _rasterize_rect(rect: RectPrimitive, cfg: MapGenCfg) -> np.ndarray:
    """Return a boolean grid marking cell centers covered by ``rect``."""
    xs, ys = _grid_centers(cfg)
    gx, gy = np.meshgrid(xs, ys)
    dx = gx - rect.center[0]
    dy = gy - rect.center[1]
    c = np.cos(rect.yaw)
    s = np.sin(rect.yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    half_x = rect.size[0] / 2.0
    half_y = rect.size[1] / 2.0
    return (np.abs(local_x) <= half_x) & (np.abs(local_y) <= half_y)


def _rasterize_pillar(pillar: PillarPrimitive, cfg: MapGenCfg) -> np.ndarray:
    """Return a boolean grid marking cell centers covered by ``pillar``."""
    xs, ys = _grid_centers(cfg)
    gx, gy = np.meshgrid(xs, ys)
    dx = gx - pillar.center[0]
    dy = gy - pillar.center[1]
    if pillar.square:
        half = pillar.radius
        return (np.abs(dx) <= half) & (np.abs(dy) <= half)

    poly = _circle_polygon(pillar.radius, pillar.segments)
    inside = np.ones(dx.shape, dtype=bool)
    n = poly.shape[0]
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = (x1 - x0) * (dy - y0) - (y1 - y0) * (dx - x0)
        inside &= cross >= 0.0
    return inside


def _rasterize_occupancy(
    rects: Sequence[RectPrimitive],
    pillars: Sequence[PillarPrimitive],
    cfg: MapGenCfg,
) -> np.ndarray:
    """Rasterize primitives only; no implicit border occupancy."""
    occupancy = np.zeros(cfg.grid_shape, dtype=np.uint8)
    for rect in rects:
        occupancy |= _rasterize_rect(rect, cfg).astype(np.uint8)
    for pillar in pillars:
        occupancy |= _rasterize_pillar(pillar, cfg).astype(np.uint8)
    return occupancy


def _inflate_occupancy(occupancy: np.ndarray, cfg: MapGenCfg) -> np.ndarray:
    """Inflate occupancy by ``cfg.inflation_cells`` cells.

    ``cfg.boundary_occupied`` now means "keep the planning border blocked in
    the *inflated* grid" -- there is no physical boundary wall in the mesh.
    """
    from scipy.ndimage import binary_dilation

    if cfg.inflation_cells <= 0:
        inflated = occupancy.copy()
    else:
        radius = int(cfg.inflation_cells)
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        inflated = binary_dilation(
            occupancy.astype(bool), structure=structure, border_value=1
        ).astype(np.uint8)
    if cfg.boundary_occupied:
        inflated[0, :] = 1
        inflated[-1, :] = 1
        inflated[:, 0] = 1
        inflated[:, -1] = 1
    return inflated


# ---------------------------------------------------------------------------
# Mesh builders (unchanged geometry contracts)
# ---------------------------------------------------------------------------


def _build_box(
    center: Tuple[float, float, float],
    size: Tuple[float, float],
    yaw: float,
    height: float,
    z_min: float = 0.0,
) -> _Mesh:
    """Build a closed, outward-wound box mesh."""
    sx, sy = size
    local = np.array(
        [
            [-sx / 2.0, -sy / 2.0, 0.0],
            [sx / 2.0, -sy / 2.0, 0.0],
            [-sx / 2.0, sy / 2.0, 0.0],
            [sx / 2.0, sy / 2.0, 0.0],
            [-sx / 2.0, -sy / 2.0, height],
            [sx / 2.0, -sy / 2.0, height],
            [-sx / 2.0, sy / 2.0, height],
            [sx / 2.0, sy / 2.0, height],
        ],
        dtype=np.float32,
    )
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = local[:, :2] @ rot.T
    vertices = np.empty_like(local)
    vertices[:, :2] = xy + np.asarray(center[:2], dtype=np.float32)
    vertices[:, 2] = local[:, 2] + center[2] + z_min

    faces = [
        (0, 2, 3, 1),  # bottom, normal -z
        (4, 5, 7, 6),  # top, normal +z
        (0, 1, 5, 4),  # -y side
        (2, 6, 7, 3),  # +y side
        (0, 4, 6, 2),  # -x side
        (1, 3, 7, 5),  # +x side
    ]
    triangles: List[Tuple[int, int, int]] = []
    for a, b, c_i, d in faces:
        triangles.append((a, b, c_i))
        triangles.append((a, c_i, d))
    return _Mesh(vertices=vertices, triangles=np.asarray(triangles, dtype=np.int32))


def _build_cylinder(
    center: Tuple[float, float],
    radius: float,
    height: float,
    segments: int,
) -> _Mesh:
    """Build a closed n-gon prism (circular pillar approximation)."""
    xy = _circle_polygon(radius, segments)
    n = xy.shape[0]
    top = np.concatenate([xy, np.full((n, 1), height)], axis=-1)
    bottom = np.concatenate([xy, np.zeros((n, 1))], axis=-1)
    center_top = np.array([[0.0, 0.0, height]], dtype=np.float32)
    center_bottom = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    vertices = np.concatenate([top, bottom, center_top, center_bottom], axis=0).astype(np.float32)
    vertices[:, :2] += np.asarray(center, dtype=np.float32)

    tris: List[Tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        tris.append((2 * n, i, j))
        tris.append((2 * n + 1, n + j, n + i))
        tris.append((i, n + j, j))
        tris.append((i, n + i, n + j))
    return _Mesh(vertices=vertices, triangles=np.asarray(tris, dtype=np.int32))


def _build_rect_mesh(rect: RectPrimitive) -> _Mesh:
    return _build_box(
        center=(rect.center[0], rect.center[1], 0.0),
        size=rect.size,
        yaw=rect.yaw,
        height=rect.height,
        z_min=0.0,
    )


def _build_pillar_mesh(pillar: PillarPrimitive) -> _Mesh:
    if pillar.square:
        return _build_box(
            center=(pillar.center[0], pillar.center[1], 0.0),
            size=(2.0 * pillar.radius, 2.0 * pillar.radius),
            yaw=0.0,
            height=pillar.height,
            z_min=0.0,
        )
    return _build_cylinder(
        center=pillar.center,
        radius=pillar.radius,
        height=pillar.height,
        segments=pillar.segments,
    )


def _build_ground_mesh(cfg: MapGenCfg) -> _Mesh:
    extent = cfg.size_m / 2.0 + cfg.ground_margin_m
    return _build_box(
        center=(0.0, 0.0, 0.0),
        size=(2.0 * extent, 2.0 * extent),
        yaw=0.0,
        height=_GROUND_THICKNESS_M,
        z_min=-_GROUND_THICKNESS_M,
    )


def _build_mesh(
    rects: Sequence[RectPrimitive],
    pillars: Sequence[PillarPrimitive],
    cfg: MapGenCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    meshes: List[_Mesh] = [_build_ground_mesh(cfg)]
    for rect in rects:
        meshes.append(_build_rect_mesh(rect))
    for pillar in pillars:
        meshes.append(_build_pillar_mesh(pillar))

    vertex_list: List[np.ndarray] = []
    triangle_list: List[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        vertex_list.append(mesh.vertices)
        triangle_list.append(mesh.triangles + offset)
        offset += mesh.vertices.shape[0]
    vertices = np.concatenate(vertex_list, axis=0).astype(np.float32)
    triangles = np.concatenate(triangle_list, axis=0).astype(np.int32)
    return vertices, triangles


# ---------------------------------------------------------------------------
# Acceptance statistics
# ---------------------------------------------------------------------------


def _estimate_near_obstacle_ratio(
    occupancy: np.ndarray,
    inflated: np.ndarray,
    cfg: MapGenCfg,
    rng: np.random.Generator,
) -> float:
    """Static proxy for the near-obstacle path statistic."""
    from scipy.ndimage import distance_transform_edt

    free_mask = occupancy == 0
    dist_to_obstacle = distance_transform_edt(
        free_mask, sampling=(cfg.resolution_m, cfg.resolution_m)
    )
    safe = (inflated == 0) & free_mask
    safe_idx = np.argwhere(safe)
    if safe_idx.shape[0] == 0:
        return 0.0
    n_sample = min(_NEAR_OBSTACLE_SAMPLE_CAP, safe_idx.shape[0])
    chosen = rng.choice(safe_idx.shape[0], size=n_sample, replace=False)
    sample_dist = dist_to_obstacle[safe_idx[chosen, 0], safe_idx[chosen, 1]]
    lo, hi = cfg.near_obstacle_range
    return float(np.mean((sample_dist >= lo) & (sample_dist <= hi)))


def _border_mask(shape: Tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _build_acceptance(
    occupancy: np.ndarray,
    inflated: np.ndarray,
    counts: dict,
    tile_types: np.ndarray,
    rects: Sequence[RectPrimitive],
    cfg: MapGenCfg,
    rng: np.random.Generator,
) -> dict:
    border = _border_mask(occupancy.shape)
    occupancy_border_free = float(not bool(np.any(occupancy[border] == 1)))
    planning_border_blocked = float(
        bool(np.all(inflated[border] == 1)) if cfg.boundary_occupied else 1.0
    )
    all_axis_aligned = float(
        all(
            min(abs(rect.yaw % (math.pi / 2.0)),
                abs(math.pi / 2.0 - (rect.yaw % (math.pi / 2.0))))
            < 1e-6
            for rect in rects
        )
    )
    has_constraint = float(
        counts[EA2_TILE_CORRIDOR] > 0 or counts[EA2_TILE_SIDE_WALLS] > 0
    )
    n_rects = len(rects)
    n_pillars = counts[EA2_TILE_PILLAR]  # tile count, not primitive count
    near_obstacle_ratio = _estimate_near_obstacle_ratio(
        occupancy, inflated, cfg, rng
    )

    # Inflated free-space connectivity: paths are sampled inside the largest
    # 8-connected component, so a map is only acceptable when that component
    # contains nearly all safe cells.
    from scipy.ndimage import label as _label
    from scipy.ndimage import sum as _nd_sum

    free = inflated == 0
    labels, n_components = _label(free)
    if n_components == 0:
        largest_free_component_ratio = 0.0
    else:
        sizes = _nd_sum(free, labels, index=range(1, n_components + 1))
        largest_free_component_ratio = float(sizes.max() / max(1, int(free.sum())))

    acceptance: dict = {
        "occupancy_border_free": occupancy_border_free,
        "planning_border_blocked": planning_border_blocked,
        "physical_boundary_walls": float(
            sum(
                1
                for r in rects
                if max(r.size) >= cfg.size_m - 0.1
            )
        ),
        "all_rects_axis_aligned": all_axis_aligned,
        "n_tiles": float(tile_types.size),
        "tile_types": float(tile_types.astype(np.float32).mean()),
        "empty_tiles": float(counts.get(EA2_TILE_EMPTY, 0)),
        "wall_tiles": float(counts.get(EA2_TILE_WALL, 0)),
        "pillar_tiles": float(counts.get(EA2_TILE_PILLAR, 0)),
        "corridor_tiles": float(counts.get(EA2_TILE_CORRIDOR, 0)),
        "side_walls_tiles": float(counts.get(EA2_TILE_SIDE_WALLS, 0)),
        "u_shape_tiles": float(counts.get(EA2_TILE_U_SHAPE, 0)),
        "has_constraint_primitive": has_constraint,
        "n_rects": float(n_rects),
        "n_pillars": float(n_pillars),
        "near_obstacle_ratio_estimate": near_obstacle_ratio,
        "near_obstacle_ratio": near_obstacle_ratio,
        "path_near_obstacle_ratio": near_obstacle_ratio,
        "largest_free_component_ratio": largest_free_component_ratio,
    }
    return acceptance


def _static_acceptance(
    occupancy: np.ndarray, acceptance: dict, cfg: MapGenCfg
) -> bool:
    if acceptance["occupancy_border_free"] != 1.0:
        return False
    if acceptance["planning_border_blocked"] != 1.0:
        return False
    if acceptance["physical_boundary_walls"] != 0.0:
        return False
    if acceptance["all_rects_axis_aligned"] != 1.0:
        return False
    if acceptance["largest_free_component_ratio"] < cfg.min_free_component_ratio:
        return False
    if cfg.require_constraint_primitive and acceptance["has_constraint_primitive"] != 1.0:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_map(cfg: MapGenCfg, seed: int) -> MapData:
    """Generate a deterministic fixed 5x5-tile map from ``cfg`` and ``seed``."""
    rng = np.random.default_rng(seed)

    for _ in range(max(1, cfg.max_gen_attempts)):
        tile_types = _make_tile_layout(rng)
        height = float(rng.uniform(*_HEIGHT_RANGE))
        rects, pillars, counts = _generate_tile_primitive_set(
            cfg, rng, tile_types, height
        )
        occupancy = _rasterize_occupancy(rects, pillars, cfg)
        inflated = _inflate_occupancy(occupancy, cfg)
        acceptance = _build_acceptance(
            occupancy, inflated, counts, tile_types, rects, cfg, rng
        )
        if _static_acceptance(occupancy, acceptance, cfg):
            break
    else:
        raise RuntimeError(
            "map_generator: failed to generate a statically acceptable map "
            f"after {cfg.max_gen_attempts} attempts"
        )

    vertices, triangles = _build_mesh(rects, pillars, cfg)
    return MapData(
        occupancy=occupancy,
        inflated=inflated,
        vertices=vertices,
        triangles=triangles,
        tile_types=tile_types,
        rects=tuple(rects),
        pillars=tuple(pillars),
        acceptance=acceptance,
    )
