"""Unit tests for the parallel A* worker glue (path_parallel)."""

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import numpy as np
import pytest

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts
from legged_gym.envs.el_4090.envelope_adaptive_2.map_generator import generate_map
from legged_gym.envs.el_4090.envelope_adaptive_2.path_parallel import (
    init_worker,
    plan_path_task,
)


def _cfg():
    return _contracts.PathCfg()


@pytest.fixture(scope="module")
def small_map():
    return generate_map(_contracts.MapGenCfg(), _contracts.PillarFieldCfg(), seed=7)


def _free_cells(inflated: np.ndarray):
    iy, ix = np.nonzero(inflated == 0)
    return ix, iy


def test_worker_without_init_raises(small_map):
    import legged_gym.envs.el_4090.envelope_adaptive_2.path_parallel as pp

    saved = (pp._OCC, pp._INFL, pp._CFG)
    pp._OCC = pp._INFL = pp._CFG = None
    try:
        with pytest.raises(RuntimeError, match="not initialized"):
            plan_path_task(0, (0.0, 0.0), (1.0, 1.0))
    finally:
        pp._OCC, pp._INFL, pp._CFG = saved


def test_plan_path_task_returns_feasible_path(small_map):
    init_worker(small_map.occupancy, small_map.inflated, _cfg())
    ix, iy = _free_cells(small_map.inflated)
    start = (-37.0 + (ix[0] + 0.5) * 0.1, -37.0 + (iy[0] + 0.5) * 0.1)
    # farthest free cell as goal
    d2 = (ix - ix[0]) ** 2 + (iy - iy[0]) ** 2
    k = int(np.argmax(d2))
    goal = (-37.0 + (ix[k] + 0.5) * 0.1, -37.0 + (iy[k] + 0.5) * 0.1)

    data = plan_path_task(0, start, goal)
    assert data.points.shape[0] >= 2
    # every path cell must be free in the inflated (planning) grid
    pix = np.floor((data.points[:, 0] + 37.0) / 0.1).astype(np.int64)
    piy = np.floor((data.points[:, 1] + 37.0) / 0.1).astype(np.int64)
    assert not small_map.inflated[piy, pix].any()
    # endpoints match the request
    assert np.allclose(data.points[0], start, atol=0.11)
    assert np.allclose(data.points[-1], goal, atol=0.11)


def test_plan_path_task_blocked_goal_raises(small_map):
    init_worker(small_map.occupancy, small_map.inflated, _cfg())
    ix, iy = _free_cells(small_map.inflated)
    start = (-37.0 + (ix[0] + 0.5) * 0.1, -37.0 + (iy[0] + 0.5) * 0.1)
    occ_iy, occ_ix = np.nonzero(small_map.occupancy == 1)
    gx = -37.0 + (occ_ix[0] + 0.5) * 0.1
    gy = -37.0 + (occ_iy[0] + 0.5) * 0.1
    with pytest.raises(ValueError):
        plan_path_task(0, start, (float(gx), float(gy)))


def test_worker_deterministic_given_seed(small_map):
    init_worker(small_map.occupancy, small_map.inflated, _cfg())
    ix, iy = _free_cells(small_map.inflated)
    start = (-37.0 + (ix[0] + 0.5) * 0.1, -37.0 + (iy[0] + 0.5) * 0.1)
    d2 = (ix - ix[0]) ** 2 + (iy - iy[0]) ** 2
    k = int(np.argmax(d2))
    goal = (-37.0 + (ix[k] + 0.5) * 0.1, -37.0 + (iy[k] + 0.5) * 0.1)
    a = plan_path_task(123, start, goal)
    b = plan_path_task(123, start, goal)
    assert a.points.shape == b.points.shape
    assert np.allclose(a.points, b.points)
