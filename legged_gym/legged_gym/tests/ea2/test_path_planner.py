"""Tests for ``path_planner.py`` (README sections 2.2.4-2.2.6).

Covered behaviours:
- straight corridor path is feasible and arc is monotonic;
- blocked goal raises a ValueError;
- noisy path points stay free in the inflated grid and segments are clear;
- heading controller converges to ``delta_target`` on a straight path;
- ego-motion decomposition formula;
- sharp synthetic path is kept feasible and its tangent yaws are smoothed to
  satisfy ``min_turn_radius``.
"""

import numpy as np
import pytest

from legged_gym.envs.el_4090.envelope_adaptive_2._contracts import PathCfg
from legged_gym.envs.el_4090.envelope_adaptive_2.path_planner import (
    _apply_path_noise,
    _astar,
    _low_pass_noise_offsets,
    _path_clear,
    _resample_polyline,
    _segment_clear,
    _simplify_path,
    ego_motion,
    heading_update,
    plan_path,
    wrap_to_pi,
)

# Map constants duplicated here to keep the test self-contained (74x74 map).
_MAP_MIN = -37.0
_RES = 0.1
_GRID = 740


def _world_to_grid(x: float, y: float) -> tuple:
    """Convert world coordinates to ``(iy, ix)`` grid indices."""
    ix = int(np.floor((x - _MAP_MIN) / _RES))
    iy = int(np.floor((y - _MAP_MIN) / _RES))
    return iy, ix


def _make_corridor(half_width: float = 0.5):
    """Create a 1.0 m wide horizontal corridor with all else occupied."""
    occ = np.zeros((_GRID, _GRID), dtype=np.uint8)
    for iy in range(_GRID):
        y = _MAP_MIN + (iy + 0.5) * _RES
        if abs(y) > half_width:
            occ[iy, :] = 1
    inflated = occ.copy()
    return occ, inflated


def _assert_points_free_and_segments_clear(points: np.ndarray, inflated: np.ndarray) -> None:
    """Assert every path point is free and every consecutive segment is clear."""
    assert points.ndim == 2 and points.shape[1] == 2
    for p in points:
        iy, ix = _world_to_grid(float(p[0]), float(p[1]))
        assert 0 <= iy < _GRID and 0 <= ix < _GRID
        assert int(inflated[iy, ix]) == 0, f"path point {p} is blocked"
    for p0, p1 in zip(points[:-1], points[1:]):
        assert _segment_clear(p0, p1, inflated), f"segment {p0} -> {p1} crosses obstacle"


def test_straight_corridor_path_feasible_and_monotonic_arc() -> None:
    """A straight corridor produces a feasible path with monotonic arc length."""
    occ, inflated = _make_corridor()
    rng = np.random.default_rng(0)
    data = plan_path(occ, inflated, (-4.0, 0.0), (4.0, 0.0), PathCfg(), rng)

    assert data.points.shape[0] > 1
    assert data.arc.shape == (data.points.shape[0],)
    assert data.yaws.shape == (data.points.shape[0],)

    assert data.arc[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(data.arc) >= -1e-9)
    assert data.arc[-1] > 7.0

    _assert_points_free_and_segments_clear(data.points, inflated)

    # The path is roughly straight along +x: tangent yaw should stay near 0.
    # Small lateral noise makes the yaw wiggle, so use a loose bound.
    assert np.all(np.abs(wrap_to_pi(data.yaws)) < 0.35)


def test_blocked_goal_raises_value_error() -> None:
    """A goal in an inflated blocked cell is rejected."""
    occ = np.zeros((_GRID, _GRID), dtype=np.uint8)
    inflated = np.zeros((_GRID, _GRID), dtype=np.uint8)
    iy, ix = _world_to_grid(2.0, 2.0)
    inflated[iy, ix] = 1

    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        plan_path(occ, inflated, (-2.0, -2.0), (2.0, 2.0), PathCfg(), rng)


def test_noisy_path_points_free_and_segments_clear() -> None:
    """Noise rejection keeps every noisy path point and segment inside free space."""
    occ, inflated = _make_corridor()
    rng = np.random.default_rng(123)
    data = plan_path(occ, inflated, (-4.0, 0.0), (4.0, 0.0), PathCfg(), rng)

    _assert_points_free_and_segments_clear(data.points, inflated)

    # The path should actually be noisy: interior lateral offsets are non-zero.
    interior_y = data.points[1:-1, 1]
    assert np.max(np.abs(interior_y)) > 1e-6


def test_arc_is_cumulative_length_of_returned_points() -> None:
    """PathData.arc must describe the returned noisy polyline, not pre-noise."""
    occ, inflated = _make_corridor()
    rng = np.random.default_rng(4)
    data = plan_path(occ, inflated, (-4.0, 0.0), (4.0, 0.0), PathCfg(), rng)

    seg_lens = np.linalg.norm(np.diff(data.points, axis=0), axis=1)
    expected_arc = np.concatenate(([0.0], np.cumsum(seg_lens)))
    assert np.allclose(data.arc, expected_arc)
    assert data.arc[-1] > 7.0


def test_min_path_len_enforced() -> None:
    """Paths shorter than cfg.min_path_len are rejected."""
    occ = np.zeros((_GRID, _GRID), dtype=np.uint8)
    inflated = np.zeros((_GRID, _GRID), dtype=np.uint8)

    with pytest.raises(ValueError, match="path length"):
        plan_path(
            occ,
            inflated,
            (-1.2, -0.8),
            (1.1, 0.9),
            PathCfg(min_path_len=20.0),
            np.random.default_rng(0),
        )


def test_apply_path_noise_fallback_stays_clear_on_narrow_corridor() -> None:
    """Direct noise application must never emit a blocked point/segment."""
    occ = np.zeros((_GRID, _GRID), dtype=np.uint8)
    for iy in range(_GRID):
        y = _MAP_MIN + (iy + 0.5) * _RES
        if abs(y) > 0.35:
            occ[iy, :] = 1
    inflated = occ.copy()
    cfg = PathCfg()
    rng = np.random.default_rng(988)

    # Build the same resampled corridor path used by plan_path.
    raw_points = _astar(inflated, (-4.0, 0.0), (4.0, 0.0))
    simple = _simplify_path(raw_points, inflated)
    resampled, _ = _resample_polyline(simple, cfg.resample_dist)

    noisy = _apply_path_noise(resampled, inflated, cfg, rng)
    assert _path_clear(noisy, inflated)


def test_heading_update_converges_to_delta_target_on_straight_path() -> None:
    """Heading tracking drives the relative yaw error to zero on a straight path."""
    tangent = 0.0
    delta_target = 0.1
    heading = 0.2
    dt = 0.02
    k_p = 5.0
    omega_max = 1.5

    for _ in range(2000):
        heading, omega, delta_actual = heading_update(
            heading,
            tangent,
            0.0,          # tangent_rate = kappa * v = 0 on a straight path
            delta_target,
            1.0,          # v
            dt,
            k_p,
            omega_max,
        )
        assert abs(omega) <= omega_max + 1e-9

    assert heading == pytest.approx(delta_target, abs=1e-3)
    assert delta_actual == pytest.approx(delta_target, abs=1e-3)


def test_ego_motion_formula_values() -> None:
    """Ego-motion is the crab-walk decomposition around the tangent frame."""
    v = 1.0
    heading = 0.2
    tangent = 0.0
    omega = 0.3
    vx, vy, out_omega = ego_motion(v, heading, tangent, omega)

    delta = heading - tangent
    assert vx == pytest.approx(v * np.cos(delta), abs=1e-12)
    assert vy == pytest.approx(v * np.sin(delta), abs=1e-12)
    assert out_omega == pytest.approx(omega, abs=1e-12)

    # A non-zero tangent shifts the body-relative decomposition.
    vx2, vy2, _ = ego_motion(2.0, 0.5, 0.2, 0.1)
    assert vx2 == pytest.approx(2.0 * np.cos(0.3), abs=1e-12)
    assert vy2 == pytest.approx(2.0 * np.sin(0.3), abs=1e-12)


def test_segment_clear_does_not_overstep_endpoint_cell() -> None:
    """DDA stops at the endpoint cell instead of visiting later blocked cells."""
    inflated = np.zeros((_GRID, _GRID), dtype=np.uint8)
    inflated[60, 99] = 1
    # The segment x-range is [3.79, 3.85], so it never enters column 99
    # (x in [3.9, 4.0)).
    assert _segment_clear((3.79, 0.0), (3.85, 0.05), inflated)


def test_plan_path_enforces_rmin_on_short_final_segment() -> None:
    """Every adjacent pair, including a short final segment, must satisfy R_min."""
    occ = np.zeros((_GRID, _GRID), dtype=np.uint8)
    inflated = np.zeros((_GRID, _GRID), dtype=np.uint8)
    cfg = PathCfg()
    data = plan_path(occ, inflated, (0.0, 0.0), (3.85, 0.05), cfg, np.random.default_rng(0))

    ds = np.linalg.norm(np.diff(data.points, axis=0), axis=1)
    dyaw = np.abs(wrap_to_pi(np.diff(data.yaws)))
    max_curvature = 1.0 / cfg.min_turn_radius
    assert np.all(ds > 0.0)
    assert np.all(dyaw / ds <= max_curvature + 1e-9)


def test_low_pass_noise_offsets_honors_configured_cutoff() -> None:
    """The configured cutoff is used directly; higher fc gives less smoothing."""
    rng_high = np.random.default_rng(123)
    rng_low = np.random.default_rng(123)
    high = _low_pass_noise_offsets(2000, 1.0, rng_high, 1.0, 0.2)
    low = _low_pass_noise_offsets(2000, 1.0, rng_low, 0.05, 0.2)

    ac_high = float(np.corrcoef(high[:-1], high[1:])[0, 1])
    ac_low = float(np.corrcoef(low[:-1], low[1:])[0, 1])
    assert ac_high < ac_low
    assert ac_low > 0.9


def test_sharp_path_yaws_smoothed_to_min_turn_radius() -> None:
    """An L-shaped path keeps clear geometry and smoothed tangent yaws."""
    occ = np.zeros((_GRID, _GRID), dtype=np.uint8)
    wall_ix, _ = _world_to_grid(0.0, 0.0)
    for iy in range(_GRID):
        y = _MAP_MIN + (iy + 0.5) * _RES
        if -5.0 <= y <= 2.0:
            occ[iy, wall_ix] = 1
    inflated = occ.copy()

    rng = np.random.default_rng(7)
    data = plan_path(occ, inflated, (-2.0, 0.0), (2.0, 0.0), PathCfg(), rng)
    ds = np.linalg.norm(np.diff(data.points, axis=0), axis=1)
    dyaw = np.abs(wrap_to_pi(np.diff(data.yaws)))
    max_curvature = 1.0 / PathCfg().min_turn_radius
    assert np.all(ds > 0.0)
    assert np.all(dyaw / ds <= max_curvature + 1e-6)
