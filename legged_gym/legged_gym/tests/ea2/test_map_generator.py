"""Unit tests for the deterministic seeded map generator (README 2.2.1-2.2.2)."""

from __future__ import annotations

import numpy as np
import pytest

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as c
from legged_gym.envs.el_4090.envelope_adaptive_2.map_generator import generate_map


def _default_cfg() -> c.MapGenCfg:
    return c.MapGenCfg()


def _circle_polygon_footprint(radius: float, segments: int) -> np.ndarray:
    """Return the CCW inscribed n-gon used for circular pillar footprints.

    This mirrors the generator's mesh polygon: a regular ``segments``-gon with
    vertices at ``radius``.  Occupancy and mesh are consistent only when both
    use this polygon rather than the exact circle.
    """
    n = max(3, int(segments))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius


def _points_in_convex_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized inclusive point-in-convex-polygon test (polygon CCW)."""
    inside = np.ones(px.shape, dtype=bool)
    n = poly.shape[0]
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
        inside &= cross >= 0.0
    return inside


def _primitive_center_coverage(map_data: c.MapData) -> np.ndarray:
    """Return a bool grid marking cell centers covered by any primitive.

    The occupancy rasterizer uses cell-center-in-footprint semantics, so this
    is the exact coverage proxy for raster-mesh consistency.  Circular pillars
    are tested against the same inscribed n-gon that the generator meshes, not
    against the exact circle.
    """
    xs = np.linspace(
        -map_data.occupancy.shape[1] / 2.0 * 0.1 + 0.05,
        map_data.occupancy.shape[1] / 2.0 * 0.1 - 0.05,
        map_data.occupancy.shape[1],
    )
    ys = np.linspace(
        -map_data.occupancy.shape[0] / 2.0 * 0.1 + 0.05,
        map_data.occupancy.shape[0] / 2.0 * 0.1 - 0.05,
        map_data.occupancy.shape[0],
    )
    gx, gy = np.meshgrid(xs, ys)
    covered = np.zeros(map_data.occupancy.shape, dtype=bool)

    for rect in map_data.rects:
        dx = gx - rect.center[0]
        dy = gy - rect.center[1]
        cos = np.cos(rect.yaw)
        sin = np.sin(rect.yaw)
        local_x = cos * dx + sin * dy
        local_y = -sin * dx + cos * dy
        covered |= (np.abs(local_x) <= rect.size[0] / 2.0) & (
            np.abs(local_y) <= rect.size[1] / 2.0
        )

    for pillar in map_data.pillars:
        dx = gx - pillar.center[0]
        dy = gy - pillar.center[1]
        if pillar.square:
            covered |= (np.abs(dx) <= pillar.radius) & (np.abs(dy) <= pillar.radius)
        else:
            poly = _circle_polygon_footprint(pillar.radius, pillar.segments)
            covered |= _points_in_convex_polygon(dx, dy, poly)
    return covered


def test_deterministic_same_seed_reproducible():
    cfg = _default_cfg()
    a = generate_map(cfg, seed=123)
    b = generate_map(cfg, seed=123)

    assert np.array_equal(a.occupancy, b.occupancy)
    assert np.array_equal(a.inflated, b.inflated)
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.triangles, b.triangles)
    assert a.rects == b.rects
    assert a.pillars == b.pillars
    assert a.acceptance == b.acceptance


def test_shapes_and_dtypes():
    map_data = generate_map(_default_cfg(), seed=7)

    assert map_data.occupancy.shape == c.EA2_GRID_SHAPE
    assert map_data.inflated.shape == c.EA2_GRID_SHAPE
    assert map_data.occupancy.dtype == np.uint8
    assert map_data.inflated.dtype == np.uint8

    assert map_data.vertices.ndim == 2
    assert map_data.vertices.shape[1] == 3
    assert map_data.vertices.dtype == np.float32
    assert map_data.vertices.shape[0] > 0

    assert map_data.triangles.ndim == 2
    assert map_data.triangles.shape[1] == 3
    assert map_data.triangles.dtype == np.int32
    assert map_data.triangles.shape[0] > 0
    assert map_data.triangles.min() >= 0
    assert map_data.triangles.max() < map_data.vertices.shape[0]

    assert isinstance(map_data.rects, tuple)
    assert isinstance(map_data.pillars, tuple)
    assert len(map_data.rects) > 0
    assert len(map_data.pillars) > 0
    assert any(p.square for p in map_data.pillars)
    assert any(not p.square for p in map_data.pillars)

    for key in (
        "boundary_occupied",
        "has_constraint_primitive",
        "near_obstacle_ratio_estimate",
        "near_obstacle_ratio",
    ):
        assert key in map_data.acceptance
        assert 0.0 <= map_data.acceptance[key] <= 1.0


def test_boundary_cells_occupied():
    occ = generate_map(_default_cfg(), seed=11).occupancy
    assert np.all(occ[0, :] == 1)
    assert np.all(occ[-1, :] == 1)
    assert np.all(occ[:, 0] == 1)
    assert np.all(occ[:, -1] == 1)


def test_inflation_monotonic_and_nontrivial():
    map_data = generate_map(_default_cfg(), seed=11)
    assert np.all(map_data.inflated >= map_data.occupancy)
    assert int((map_data.inflated - map_data.occupancy).sum()) > 0


def test_mesh_triangle_count_and_watertight_edges():
    map_data = generate_map(_default_cfg(), seed=42)
    triangles = map_data.triangles
    assert triangles.shape[0] >= 12  # at least a ground slab
    # A closed triangle soup has every undirected edge appearing exactly twice.
    edge_counts: dict = {}
    for a, b, cc in triangles:
        for e in ((int(a), int(b)), (int(b), int(cc)), (int(cc), int(a))):
            key = (min(e), max(e))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    assert len(edge_counts) > 0
    assert all(count == 2 for count in edge_counts.values())


def test_mesh_winding_is_outward_consistent():
    map_data = generate_map(_default_cfg(), seed=42)
    vertices = map_data.vertices
    triangles = map_data.triangles

    parent = list(range(vertices.shape[0]))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for tri in triangles:
        union(int(tri[0]), int(tri[1]))
        union(int(tri[1]), int(tri[2]))

    components: dict = {}
    for v in range(vertices.shape[0]):
        root = find(v)
        components.setdefault(root, []).append(v)

    for comp_vertices in components.values():
        comp_set = set(comp_vertices)
        centroid = vertices[comp_vertices].mean(axis=0)
        for tri in triangles:
            if int(tri[0]) not in comp_set:
                continue
            v0 = vertices[int(tri[0])]
            v1 = vertices[int(tri[1])]
            v2 = vertices[int(tri[2])]
            normal = np.cross(v1 - v0, v2 - v0)
            tri_center = (v0 + v1 + v2) / 3.0
            # For every convex closed component, an outward face has its
            # normal pointing away from the component's vertex centroid.
            assert np.dot(normal, tri_center - centroid) > -1e-5


def test_circular_pillar_rasterization_matches_polygon_footprint():
    map_data = generate_map(_default_cfg(), seed=0)
    xs = np.linspace(-5.95, 5.95, map_data.occupancy.shape[1])
    ys = np.linspace(-5.95, 5.95, map_data.occupancy.shape[0])
    gx, gy = np.meshgrid(xs, ys)

    for pillar in map_data.pillars:
        if pillar.square:
            continue
        dx = gx - pillar.center[0]
        dy = gy - pillar.center[1]
        poly = _circle_polygon_footprint(pillar.radius, pillar.segments)
        polygon_covered = _points_in_convex_polygon(dx, dy, poly)
        # A circular pillar's own footprint must not cover any cell center
        # that the generator's n-gon mesh would not cover.
        # We cannot isolate the pillar in the combined occupancy, so verify
        # against the generator's private rasterizer output.
        from legged_gym.envs.el_4090.envelope_adaptive_2.map_generator import (
            _rasterize_pillar,
        )

        rasterized = _rasterize_pillar(pillar, _default_cfg()).astype(bool)
        assert bool(np.all(rasterized == polygon_covered))


def test_occupied_cell_centers_covered_by_primitives():
    # Seed 0 was the concrete review regression: exact-circle rasterization
    # produced occupied cells outside the 16-gon mesh footprint.
    map_data = generate_map(_default_cfg(), seed=0)
    covered = _primitive_center_coverage(map_data)
    assert bool(np.all(covered[map_data.occupancy == 1]))


def test_acceptance_requires_constraint_primitive():
    map_data = generate_map(_default_cfg(), seed=3)
    assert map_data.acceptance["boundary_occupied"] == 1.0
    assert map_data.acceptance["has_constraint_primitive"] == 1.0
    assert map_data.acceptance["corridor_walls"] >= 2.0
    assert map_data.acceptance["side_wall_segments"] >= 2.0
    assert map_data.acceptance["u_walls"] == 3.0
