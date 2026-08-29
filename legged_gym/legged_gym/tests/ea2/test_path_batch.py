"""Tests for ``path_batch.PathBatch`` (Block 1 data layer)."""

from __future__ import annotations

import math

import isaacgym  # noqa: F401 -- must be imported before torch.
import numpy as np
import pytest
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2._contracts import PathData
from legged_gym.envs.el_4090.envelope_adaptive_2.path_batch import (
    DEFAULT_MAX_CORNERS,
    DEFAULT_MAX_POINTS,
    PathBatch,
)


def _make_path(
    n=20,
    with_seg=True,
    with_corners=True,
    seed=0,
):
    rng = np.random.default_rng(seed)
    pts = [np.array([0.0, 0.0])]
    h = 0.0
    for _ in range(n - 1):
        h += rng.uniform(-0.2, 0.2)
        step = rng.uniform(0.3, 0.6)
        pts.append(pts[-1] + step * np.array([math.cos(h), math.sin(h)]))
    pts = np.asarray(pts, dtype=np.float64)

    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg_len)))
    yaws = np.arctan2(np.diff(pts, axis=0)[:, 1], np.diff(pts, axis=0)[:, 0])
    yaws = np.concatenate((yaws, yaws[-1:]))

    if with_seg:
        seg_dirs = np.arctan2(
            np.diff(pts, axis=0)[:, 1], np.diff(pts, axis=0)[:, 0]
        )
    else:
        seg_dirs = None

    if with_corners:
        # Pick every 4th interior vertex as a coarse corner set.
        corner_arcs = arc[3:-1:4]
        corner_targets = yaws[3:-1:4]
    else:
        corner_arcs = None
        corner_targets = None

    return PathData(
        points=pts,
        yaws=yaws,
        arc=arc,
        segment_dirs=seg_dirs,
        corner_arcs=corner_arcs,
        corner_targets=corner_targets,
    )


def _pb(num=4, max_points=None, max_corners=None, device="cpu"):
    return PathBatch(
        num_envs=num,
        max_points=max_points or DEFAULT_MAX_POINTS,
        max_corners=max_corners or DEFAULT_MAX_CORNERS,
        device=device,
    )


def test_install_basic_and_invariants():
    pb = _pb()
    path = _make_path()
    pb.install(0, path)
    assert bool(pb.valid[0])
    assert int(pb.lengths[0]) == len(path.points)
    assert bool(pb.has_seg[0])
    assert int(pb.corner_lengths[0]) > 0
    pb.assert_invariants()

    # Points/arc mirror the source.
    torch.testing.assert_close(
        pb.points[0, : len(path.points)],
        torch.from_numpy(path.points).float(),
    )
    torch.testing.assert_close(
        pb.arc[0, : len(path.arc)],
        torch.from_numpy(path.arc).float(),
    )
    torch.testing.assert_close(
        pb.yaws[0, : len(path.yaws)],
        torch.from_numpy(path.yaws).float(),
    )


def test_arc_and_corner_tail_are_inf():
    pb = _pb()
    path = _make_path()
    pb.install(0, path)
    n = len(path.points)
    k = len(path.corner_arcs)
    assert bool(torch.isinf(pb.arc[0, n:]).all())
    assert bool(torch.isinf(pb.corner_arcs[0, k:]).all())
    assert bool(torch.isfinite(pb.arc[0, :n]).all())
    assert bool(torch.isfinite(pb.corner_arcs[0, :k]).all())


def test_install_none_invalidates():
    pb = _pb()
    pb.install(0, None)
    assert not bool(pb.valid[0])
    assert int(pb.lengths[0]) == 0
    assert int(pb.corner_lengths[0]) == 0
    assert not bool(pb.has_seg[0])


def test_install_without_segment_and_corner():
    pb = _pb()
    path = _make_path(with_seg=False, with_corners=False)
    pb.install(0, path)
    assert bool(pb.valid[0])
    assert not bool(pb.has_seg[0])
    assert int(pb.corner_lengths[0]) == 0
    assert bool(torch.isinf(pb.arc[0, len(path.points):]).all())
    pb.assert_invariants()


def test_install_twice_is_idempotent():
    pb = _pb()
    p1 = _make_path(seed=1, with_corners=True)
    p2 = _make_path(seed=2, with_seg=False, with_corners=False)
    pb.install(0, p1)
    pb.install(0, p2)
    assert int(pb.lengths[0]) == len(p2.points)
    assert int(pb.corner_lengths[0]) == 0
    assert not bool(pb.has_seg[0])  # p2 has no segment metadata in this helper
    torch.testing.assert_close(
        pb.points[0, : len(p2.points)],
        torch.from_numpy(p2.points).float(),
    )


def test_ensure_capacity_preserves_old_rows():
    pb = _pb(max_points=32, max_corners=8)
    path = _make_path(n=20)
    pb.install(0, path)
    expected_points = pb.points[0, : len(path.points)].clone()

    long_path = _make_path(n=90, with_corners=True)
    pb.install(1, long_path)

    assert pb.max_points > 32
    assert pb.max_corners > 8
    torch.testing.assert_close(
        pb.points[0, : len(path.points)],
        expected_points,
    )
    assert int(pb.lengths[1]) == len(long_path.points)
    pb.assert_invariants()


def test_assert_invariants_detects_corruption():
    pb = _pb()
    path = _make_path()
    pb.install(0, path)
    pb.arc[0, 0] = float("nan")
    with pytest.raises(AssertionError):
        pb.assert_invariants()


def _reference_interpolate_path(path, s):
    """Reference interpolator mirroring the removed env helper.

    Kept here as the independent oracle for ``PathBatch.query``; production
    consumes PathBatch directly.
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
    dyaw = float((yaws[idx + 1] - yaws[idx] + np.pi) % (2.0 * np.pi) - np.pi)
    tangent_rate = dyaw / ds
    return xy, tangent, tangent_rate


def test_query_matches_interpolate_and_segment_dirs():
    paths = [_make_path(seed=i) for i in range(4)]
    pb = _pb()
    for i, p in enumerate(paths):
        pb.install(i, p)

    rng = np.random.default_rng(7)
    for _ in range(100):
        s_all = torch.zeros(4, dtype=torch.float32)
        s_values = []
        for i, p in enumerate(paths):
            s = float(rng.uniform(p.arc[0] + 1e-3, p.arc[-1] - 1e-3))
            s_all[i] = s
            s_values.append(s)

        res = pb.query(s_all)
        for i, p in enumerate(paths):
            s = s_values[i]
            xy_ref, _, _ = _reference_interpolate_path(p, s)
            si = int(np.searchsorted(p.arc, s, side="right") - 1)
            si = min(max(si, 0), len(p.segment_dirs) - 1)
            seg_ref = float(p.segment_dirs[si])

            torch.testing.assert_close(
                res.xy[i],
                torch.tensor(xy_ref, dtype=torch.float32),
                atol=1e-5,
                rtol=1e-5,
            )
            assert abs(float(res.tangent[i]) - seg_ref) < 1e-5

            ci = int(np.searchsorted(p.corner_arcs, s, side="right"))
            if ci < len(p.corner_arcs):
                assert bool(res.has_next_corner[i])
                assert (
                    abs(float(res.next_corner[i]) - float(p.corner_arcs[ci]))
                    < 1e-5
                )
            else:
                assert not bool(res.has_next_corner[i])


def test_query_fallback_to_yaws_without_segment():
    pb = _pb()
    p = _make_path(with_seg=False, with_corners=False)
    pb.install(0, p)

    s = float((p.arc[0] + p.arc[-1]) / 2.0)
    res = pb.query(torch.tensor([s], dtype=torch.float32))
    _, tangent_ref, _ = _reference_interpolate_path(p, s)
    assert abs(float(res.tangent[0]) - tangent_ref) < 1e-5


def test_query_invalid_row_is_safe():
    pb = _pb(num=2)
    pb.install(0, _make_path(seed=0))
    pb.install(1, None)
    res = pb.query(torch.zeros(2, dtype=torch.float32))
    assert res.xy.shape == (2, 2)
    assert not bool(res.has_next_corner[1])
    assert bool(torch.isfinite(res.xy).all())


def test_query_no_corner_sets_has_next_false():
    pb = _pb()
    pb.install(0, _make_path(with_corners=False, with_seg=False))
    res = pb.query(torch.zeros(1, dtype=torch.float32))
    assert not bool(res.has_next_corner[0])
