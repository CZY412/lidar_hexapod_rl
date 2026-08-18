"""Deterministic seeded map generator for envelope_adaptive_2 (M1).

This module implements the README v2 ``map_generator.py`` contract:
primitive obstacles -> occupancy grid -> inflated safety grid -> Warp mesh.
The occupancy grid is the authoritative geometry; the Warp mesh is built from
the same primitives (boxes and n-gon cylinders) so rasterization and mesh stay
consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ._contracts import (
    MapData,
    MapGenCfg,
    PillarPrimitive,
    RectPrimitive,
)

# Default obstacle ranges mirror the frozen ``El4090EA2Cfg.obstacles`` block.
_HEIGHT_RANGE = (1.5, 2.0)
_WALL_LENGTH_RANGE = (2.0, 5.0)
_WALL_THICKNESS_RANGE = (0.2, 0.5)
_PILLAR_HALF_RANGE = (0.2, 0.4)
_PILLAR_SEGMENTS = 16
_CORRIDOR_WIDTH_RANGE = (1.0, 2.0)
_SIDE_WALL_COUNT_RANGE = (2, 5)
_U_OPENING_RANGE = (1.0, 1.5)

_GROUND_THICKNESS_M = 0.02
_BOUNDARY_WALL_THICKNESS_M = 0.2
_NEAR_OBSTACLE_SAMPLE_CAP = 2000


@dataclass(frozen=True)
class _Mesh:
    """Small internal mesh container used while assembling the final arrays."""

    vertices: np.ndarray
    triangles: np.ndarray


def _world_to_grid(x: float, y: float, cfg: MapGenCfg) -> Tuple[int, int]:
    """Convert a world coordinate to grid indices using the contract formula.

    ``ix = floor((x + size/2) / resolution)``, ``iy`` likewise.  The caller is
    responsible for bounds checking.
    """
    half = cfg.size_m / 2.0
    ix = int(np.floor((x + half) / cfg.resolution_m))
    iy = int(np.floor((y + half) / cfg.resolution_m))
    return ix, iy


def _grid_centers(cfg: MapGenCfg) -> Tuple[np.ndarray, np.ndarray]:
    """Return world coordinates of every grid-cell center.

    Returns ``(xs, ys)`` with shapes ``(grid_shape[1],)`` and
    ``(grid_shape[0],)``.  The grid rows correspond to ``iy`` and columns to
    ``ix``.
    """
    world_min = -cfg.size_m / 2.0
    xs = world_min + (np.arange(cfg.grid_shape[1]) + 0.5) * cfg.resolution_m
    ys = world_min + (np.arange(cfg.grid_shape[0]) + 0.5) * cfg.resolution_m
    return xs, ys


def _circle_polygon(radius: float, segments: int) -> np.ndarray:
    """Return the CCW inscribed n-gon used for circular pillar footprints.

    The same vertex set is used by both the occupancy rasterizer and the
    n-gon prism mesh so the authoritative grid and the raycast mesh describe
    exactly the same 2D footprint.
    """
    n = max(3, int(segments))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius


def _sample_center(rng: np.random.Generator, limit: float) -> Tuple[float, float]:
    """Sample a 2D center uniformly inside ``[-limit, limit]^2``."""
    x = float(rng.uniform(-limit, limit))
    y = float(rng.uniform(-limit, limit))
    return x, y


def _generate_boundary_rects(cfg: MapGenCfg, height: float) -> List[RectPrimitive]:
    """Build four thin boundary walls so the map border is explicitly occupied.

    These rects are first-class primitives: they appear in ``MapData.rects``,
    are rasterized into occupancy, and are meshed as normal wall boxes.
    """
    half = cfg.size_m / 2.0
    t = _BOUNDARY_WALL_THICKNESS_M
    return [
        RectPrimitive(center=(0.0, -half + t / 2.0), size=(cfg.size_m, t), yaw=0.0, height=height),
        RectPrimitive(center=(0.0, half - t / 2.0), size=(cfg.size_m, t), yaw=0.0, height=height),
        RectPrimitive(center=(-half + t / 2.0, 0.0), size=(t, cfg.size_m), yaw=0.0, height=height),
        RectPrimitive(center=(half - t / 2.0, 0.0), size=(t, cfg.size_m), yaw=0.0, height=height),
    ]


def _generate_corridor(rng: np.random.Generator, height: float) -> List[RectPrimitive]:
    """Generate one narrow corridor: two parallel wall segments with a gap.

    The gap is sampled from the README's 1.0-2.0 m range and is guaranteed by
    construction: wall centers are offset by ``gap/2 + thickness/2`` on each
    side of the corridor centerline.
    """
    length = float(rng.uniform(3.0, 5.0))
    thickness = float(rng.uniform(*_WALL_THICKNESS_RANGE))
    gap = float(rng.uniform(*_CORRIDOR_WIDTH_RANGE))
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    cx, cy = _sample_center(rng, 2.0)
    normal_x = -np.sin(yaw)
    normal_y = np.cos(yaw)
    offset = gap / 2.0 + thickness / 2.0
    return [
        RectPrimitive(
            center=(cx + normal_x * offset, cy + normal_y * offset),
            size=(length, thickness),
            yaw=yaw,
            height=height,
        ),
        RectPrimitive(
            center=(cx - normal_x * offset, cy - normal_y * offset),
            size=(length, thickness),
            yaw=yaw,
            height=height,
        ),
    ]


def _generate_side_wall_group(rng: np.random.Generator, height: float) -> List[RectPrimitive]:
    """Generate one side-wall group: 2-5 wall segments on one side of a line."""
    n_segments = int(rng.integers(*_SIDE_WALL_COUNT_RANGE))
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    base_x, base_y = _sample_center(rng, 2.0)
    dir_x, dir_y = np.cos(yaw), np.sin(yaw)
    normal_x, normal_y = -dir_y, dir_x
    lateral = float(rng.uniform(0.8, 1.5))
    along = rng.uniform(-2.0, 2.0, size=n_segments)

    rects: List[RectPrimitive] = []
    for s in along:
        seg_len = float(rng.uniform(1.5, 3.0))
        seg_thick = float(rng.uniform(*_WALL_THICKNESS_RANGE))
        center = (
            base_x + dir_x * float(s) + normal_x * lateral,
            base_y + dir_y * float(s) + normal_y * lateral,
        )
        rects.append(
            RectPrimitive(
                center=center,
                size=(seg_len, seg_thick),
                yaw=yaw,
                height=height,
            )
        )
    return rects


def _generate_u_shape(rng: np.random.Generator, height: float) -> List[RectPrimitive]:
    """Generate one U-shape: a back wall plus two parallel side walls.

    The opening width is sampled from 1.0-1.5 m and faces the local +x
    direction of the sampled yaw.
    """
    depth = float(rng.uniform(2.0, 3.0))
    opening = float(rng.uniform(*_U_OPENING_RANGE))
    thickness = float(rng.uniform(*_WALL_THICKNESS_RANGE))
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    base_x, base_y = _sample_center(rng, 2.0)
    dir_x, dir_y = np.cos(yaw), np.sin(yaw)
    normal_x, normal_y = -dir_y, dir_x

    back_center = (base_x + dir_x * (-depth / 2.0), base_y + dir_y * (-depth / 2.0))
    back = RectPrimitive(
        center=back_center,
        size=(opening + 2.0 * thickness, thickness),
        yaw=yaw,
        height=height,
    )

    side_offset = opening / 2.0 + thickness / 2.0
    side1 = RectPrimitive(
        center=(base_x + normal_x * side_offset, base_y + normal_y * side_offset),
        size=(depth, thickness),
        yaw=yaw,
        height=height,
    )
    side2 = RectPrimitive(
        center=(base_x - normal_x * side_offset, base_y - normal_y * side_offset),
        size=(depth, thickness),
        yaw=yaw,
        height=height,
    )
    return [back, side1, side2]


def _generate_random_walls(rng: np.random.Generator, height: float) -> List[RectPrimitive]:
    """Generate a handful of independent wall primitives."""
    n = int(rng.integers(1, 4))
    rects: List[RectPrimitive] = []
    for _ in range(n):
        length = float(rng.uniform(2.0, 4.0))
        thickness = float(rng.uniform(*_WALL_THICKNESS_RANGE))
        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        cx, cy = _sample_center(rng, 3.0)
        rects.append(
            RectPrimitive(
                center=(cx, cy),
                size=(length, thickness),
                yaw=yaw,
                height=height,
            )
        )
    return rects


def _generate_pillars(rng: np.random.Generator, height: float) -> List[PillarPrimitive]:
    """Generate 4-8 square/circular pillar primitives.

    At least one square and one circular pillar are always produced so a map
    exercises both mesh code paths.
    """
    n = int(rng.integers(4, 9))
    square_flags = [True, False] + [bool(rng.random() < 0.5) for _ in range(n - 2)]
    rng.shuffle(square_flags)
    pillars: List[PillarPrimitive] = []
    for square in square_flags:
        radius = float(rng.uniform(*_PILLAR_HALF_RANGE))
        pillar_height = float(rng.uniform(*_HEIGHT_RANGE))
        cx, cy = _sample_center(rng, 4.5)
        pillars.append(
            PillarPrimitive(
                center=(cx, cy),
                radius=radius,
                height=pillar_height,
                square=square,
                segments=_PILLAR_SEGMENTS,
            )
        )
    return pillars


def _generate_primitive_set(
    cfg: MapGenCfg, rng: np.random.Generator
) -> Tuple[List[RectPrimitive], List[PillarPrimitive], dict]:
    """Generate the complete primitive set for one map.

    Returns ``(rects, pillars, counts)`` where ``counts`` records how many
    corridor / side-wall / U-shape wall primitives were produced.
    """
    height = float(rng.uniform(*_HEIGHT_RANGE))
    rects: List[RectPrimitive] = _generate_boundary_rects(cfg, height)
    corridor_rects = _generate_corridor(rng, height)
    side_rects = _generate_side_wall_group(rng, height)
    u_rects = _generate_u_shape(rng, height)
    random_rects = _generate_random_walls(rng, height)
    pillars = _generate_pillars(rng, height)

    rects.extend(corridor_rects)
    rects.extend(side_rects)
    rects.extend(u_rects)
    rects.extend(random_rects)

    counts = {
        "corridor_walls": len(corridor_rects),
        "side_wall_segments": len(side_rects),
        "u_walls": len(u_rects),
        "random_walls": len(random_rects),
        "pillars": len(pillars),
    }
    return rects, pillars, counts


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
    """Return a boolean grid marking cell centers covered by ``pillar``.

    Circular pillars use the same inscribed n-gon footprint as
    ``_build_cylinder``: the mesh polygon is authoritative for both occupancy
    and rasterization, so no cell is marked occupied that lies outside the
    generated raycast mesh footprint.
    """
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
    """Rasterize all primitives into the authoritative occupancy grid."""
    occupancy = np.zeros(cfg.grid_shape, dtype=np.uint8)
    for rect in rects:
        occupancy |= _rasterize_rect(rect, cfg).astype(np.uint8)
    for pillar in pillars:
        occupancy |= _rasterize_pillar(pillar, cfg).astype(np.uint8)
    # Belt-and-braces: boundary is always occupied regardless of rasterization.
    occupancy[0, :] = 1
    occupancy[-1, :] = 1
    occupancy[:, 0] = 1
    occupancy[:, -1] = 1
    return occupancy


def _inflate_occupancy(occupancy: np.ndarray, cfg: MapGenCfg) -> np.ndarray:
    """Inflate the occupancy grid by ``cfg.inflation_cells`` cells.

    A square structuring element implements the 8-neighbour 4-cell safety
    dilation used by A* and path-noise rejection.
    """
    from scipy.ndimage import binary_dilation

    if cfg.inflation_cells <= 0:
        return occupancy.copy()
    radius = int(cfg.inflation_cells)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    inflated = binary_dilation(
        occupancy.astype(bool), structure=structure, border_value=1
    ).astype(np.uint8)
    # Boundary stays blocked after dilation.
    inflated[0, :] = 1
    inflated[-1, :] = 1
    inflated[:, 0] = 1
    inflated[:, -1] = 1
    return inflated


def _build_box(
    center: Tuple[float, float, float],
    size: Tuple[float, float],
    yaw: float,
    height: float,
    z_min: float = 0.0,
) -> _Mesh:
    """Build a closed, outward-wound box mesh.

    ``size`` is ``(length_x, length_y)`` in the box's local frame.  The box
    spans ``z_min .. z_min + height`` and is rotated by ``yaw`` around z.
    """
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

    # Outward-facing quad vertex orders (verified in tests by edge/watertight
    # and winding conventions).  The bottom quad is CW from +z so both of its
    # triangles have normal -z (outward through the bottom of the slab/box).
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
    """Build a closed n-gon prism (circular pillar approximation).

    The polygon is oriented CCW when viewed from +z.  Top, bottom and side
    faces all use outward winding.
    """
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
        # Top cap, normal +z.
        tris.append((2 * n, i, j))
        # Bottom cap, normal -z.
        tris.append((2 * n + 1, n + j, n + i))
        # Side quad, outward normal.
        tris.append((i, n + j, j))
        tris.append((i, n + i, n + j))
    return _Mesh(vertices=vertices, triangles=np.asarray(tris, dtype=np.int32))


def _build_rect_mesh(rect: RectPrimitive) -> _Mesh:
    """Build a closed box mesh for a ``RectPrimitive`` wall."""
    return _build_box(
        center=(rect.center[0], rect.center[1], 0.0),
        size=rect.size,
        yaw=rect.yaw,
        height=rect.height,
        z_min=0.0,
    )


def _build_pillar_mesh(pillar: PillarPrimitive) -> _Mesh:
    """Build a box or n-gon prism mesh for a ``PillarPrimitive``."""
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
    """Build the ground as a thin closed box whose top surface is z=0.

    The ground covers the map plus ``cfg.ground_margin_m`` on every side.  A
    thin closed slab is used so the combined triangle soup is watertight.
    """
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
    """Assemble the combined ground + obstacle mesh arrays."""
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


def _estimate_near_obstacle_ratio(
    occupancy: np.ndarray,
    inflated: np.ndarray,
    cfg: MapGenCfg,
    rng: np.random.Generator,
) -> float:
    """Estimate the near-obstacle path statistic without running A*.

    The A* validation in the environment samples actual paths.  This static
    proxy samples safe (inflated-free) cells and computes their distance to
    the nearest occupied cell using an EDT, then reports the fraction whose
    distance falls inside ``cfg.near_obstacle_range``.
    """
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


def _build_acceptance(
    occupancy: np.ndarray,
    inflated: np.ndarray,
    counts: dict,
    cfg: MapGenCfg,
    rng: np.random.Generator,
) -> dict:
    """Build the static acceptance statistics stored in ``MapData.acceptance``."""
    border = np.ones(occupancy.shape, dtype=bool)
    border[1:-1, 1:-1] = False
    boundary_ok = float(np.all(occupancy[border] == 1))
    has_constraint = float((counts["corridor_walls"] > 0) or (counts["side_wall_segments"] > 0))
    near_obstacle_ratio = _estimate_near_obstacle_ratio(occupancy, inflated, cfg, rng)

    acceptance: dict = {
        "boundary_occupied": boundary_ok,
        "has_corridor_or_side_wall": has_constraint,
        "has_constraint_primitive": has_constraint,
        "n_rects": float(counts.get("n_rects", 0.0)),
        "n_pillars": float(counts.get("pillars", 0.0)),
        "corridor_walls": float(counts.get("corridor_walls", 0.0)),
        "side_wall_segments": float(counts.get("side_wall_segments", 0.0)),
        "u_walls": float(counts.get("u_walls", 0.0)),
        "near_obstacle_ratio_estimate": near_obstacle_ratio,
        "near_obstacle_ratio": near_obstacle_ratio,
        "path_near_obstacle_ratio": near_obstacle_ratio,
    }
    return acceptance


def _static_acceptance(occupancy: np.ndarray, acceptance: dict, cfg: MapGenCfg) -> bool:
    """Return True when static map acceptance criteria are satisfied."""
    if acceptance["boundary_occupied"] != 1.0:
        return False
    if cfg.require_constraint_primitive and acceptance["has_corridor_or_side_wall"] != 1.0:
        return False
    return True


def generate_map(cfg: MapGenCfg, seed: int) -> MapData:
    """Generate a deterministic fixed map from ``cfg`` and ``seed``.

    The returned ``MapData`` contains the occupancy grid, the 4-cell inflated
    safety grid, a watertight ground+obstacle mesh, the primitive lists, and
    static acceptance statistics.
    """
    rng = np.random.default_rng(seed)

    for _ in range(max(1, cfg.max_gen_attempts)):
        rects, pillars, counts = _generate_primitive_set(cfg, rng)
        counts["n_rects"] = len(rects)
        occupancy = _rasterize_occupancy(rects, pillars, cfg)
        inflated = _inflate_occupancy(occupancy, cfg)
        acceptance = _build_acceptance(occupancy, inflated, counts, cfg, rng)
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
        rects=tuple(rects),
        pillars=tuple(pillars),
        acceptance=acceptance,
    )
