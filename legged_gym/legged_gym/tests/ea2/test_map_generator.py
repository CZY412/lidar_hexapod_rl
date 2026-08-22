"""Unit tests for the pd_gru pillar-field map generator."""

from __future__ import annotations

import numpy as np

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as c
from legged_gym.envs.el_4090.envelope_adaptive_2.map_generator import (
    generate_map,
)


def _cfg() -> c.MapGenCfg:
    return c.MapGenCfg()


def _pillar_cfg() -> c.PillarFieldCfg:
    return c.PillarFieldCfg()


def _primitive_center_coverage(map_data: c.MapData) -> np.ndarray:
    """Cell centers covered by the primitive footprints (cell-center rule)."""
    shape = map_data.occupancy.shape
    half = c.EA2_MAP_SIZE_M / 2.0
    xs = np.linspace(-half + 0.05, half - 0.05, shape[1])
    ys = np.linspace(-half + 0.05, half - 0.05, shape[0])
    gx, gy = np.meshgrid(xs, ys)
    covered = np.zeros(shape, dtype=bool)
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
    return covered


def test_deterministic_same_seed_reproducible():
    a = generate_map(_cfg(), _pillar_cfg(), seed=123)
    b = generate_map(_cfg(), _pillar_cfg(), seed=123)
    assert np.array_equal(a.occupancy, b.occupancy)
    assert np.array_equal(a.inflated, b.inflated)
    assert np.array_equal(a.distance_field, b.distance_field)
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.triangles, b.triangles)
    assert a.rects == b.rects
    assert a.acceptance == b.acceptance


def test_shapes_and_dtypes():
    m = generate_map(_cfg(), _pillar_cfg(), seed=0)
    assert m.occupancy.shape == c.EA2_GRID_SHAPE
    assert m.inflated.shape == c.EA2_GRID_SHAPE
    assert m.distance_field is not None
    assert m.distance_field.shape == c.EA2_GRID_SHAPE
    assert m.distance_field.dtype == np.float32
    assert m.occupancy.dtype == np.uint8
    assert m.inflated.dtype == np.uint8
    assert m.vertices.ndim == 2 and m.vertices.shape[1] == 3
    assert m.vertices.dtype == np.float32
    assert m.triangles.ndim == 2 and m.triangles.shape[1] == 3
    assert m.triangles.dtype == np.int32
    assert len(m.rects) > 0
    assert len(m.pillars) == 0


def test_no_physical_boundary_walls_planning_border_blocked():
    m = generate_map(_cfg(), _pillar_cfg(), seed=11)
    occ = m.occupancy
    inf = m.inflated
    assert np.all(occ[0, :] == 0)
    assert np.all(occ[-1, :] == 0)
    assert np.all(occ[:, 0] == 0)
    assert np.all(occ[:, -1] == 0)
    assert np.all(inf[0, :] == 1)
    assert np.all(inf[-1, :] == 1)
    assert np.all(inf[:, 0] == 1)
    assert np.all(inf[:, -1] == 1)
    assert m.acceptance["physical_boundary_walls"] == 0.0


def test_tile_layout_is_4x4_pillar_field():
    m = generate_map(_cfg(), _pillar_cfg(), seed=7)
    assert m.tile_types is not None
    assert m.tile_types.shape == (4, 4)
    assert np.all(m.tile_types == c.EA2_TILE_PILLAR)
    assert m.acceptance["n_tiles"] == 16.0


def test_all_rects_axis_aligned():
    m = generate_map(_cfg(), _pillar_cfg(), seed=7)
    assert len(m.rects) > 0
    for rect in m.rects:
        assert abs(rect.yaw % (np.pi / 2.0)) < 1e-6


def test_inflated_free_space_largest_component_dominates():
    for seed in range(10):
        m = generate_map(_cfg(), _pillar_cfg(), seed=seed)
        assert m.acceptance["largest_free_component_ratio"] >= 0.95


def test_mesh_watertight_and_outward_winding():
    m = generate_map(_cfg(), _pillar_cfg(), seed=42)
    triangles = m.triangles
    edge_counts = {}
    for a, b, cc in triangles:
        for e in ((int(a), int(b)), (int(b), int(cc)), (int(cc), int(a))):
            key = (min(e), max(e))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    assert all(count == 2 for count in edge_counts.values())

    vertices = m.vertices
    parent = list(range(vertices.shape[0]))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for tri in triangles:
        union(int(tri[0]), int(tri[1]))
        union(int(tri[1]), int(tri[2]))

    components = {}
    for v in range(vertices.shape[0]):
        components.setdefault(find(v), []).append(v)
    for comp in components.values():
        comp_set = set(comp)
        centroid = vertices[comp].mean(axis=0)
        for tri in triangles:
            if int(tri[0]) not in comp_set:
                continue
            v0, v1, v2 = vertices[int(tri[0])], vertices[int(tri[1])], vertices[int(tri[2])]
            normal = np.cross(v1 - v0, v2 - v0)
            tri_center = (v0 + v1 + v2) / 3.0
            assert np.dot(normal, tri_center - centroid) > -1e-5


def test_occupied_cell_centers_covered_by_primitives():
    m = generate_map(_cfg(), _pillar_cfg(), seed=0)
    covered = _primitive_center_coverage(m)
    assert bool(np.all(covered[m.occupancy == 1]))


def test_acceptance_keys():
    m = generate_map(_cfg(), _pillar_cfg(), seed=3)
    for key in (
        "occupancy_border_free",
        "planning_border_blocked",
        "physical_boundary_walls",
        "all_rects_axis_aligned",
        "largest_free_component_ratio",
        "n_tiles",
        "n_rects",
        "tiles_with_pillars",
        "near_obstacle_ratio",
    ):
        assert key in m.acceptance
