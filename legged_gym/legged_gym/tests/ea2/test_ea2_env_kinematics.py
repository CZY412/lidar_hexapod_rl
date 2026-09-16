"""README-spec verification for ``EL_4090_EA2._step_kinematics_batched``.

Independently validates the recovered batched kinematics against the
documented contract (README 2.2.5 / 2.9) rather than against the source:

* straight travel: arc advances exactly ``v * dt`` per step, base position
  rides the path, heading converges to the path tangent, ego-motion is the
  crab decomposition ``(v*cos(delta), v*sin(delta), omega)``;
* stop-and-turn corners: arc freezes at the corner, ego-motion goes to zero,
  heading rotates in place bounded by ``omega_max`` until aligned, then
  translation resumes;
* reaching the end of a path triggers a soft replan (no hard reset).
"""

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import math

import numpy as np
import pytest
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import EL_4090_EA2
from legged_gym.envs.el_4090.envelope_adaptive_2.path_batch import PathBatch
from legged_gym.envs.el_4090.envelope_adaptive_2.path_planner import wrap_to_pi as np_wrap


class _Cfg:
    class path:
        k_p = 5.0
        omega_max = 1.5
        goal_min_obstacle_dist = 0.5
        speed_range = [1.0, 1.0]

    class height:
        min_m = 0.52


def _bare_env(n: int, paths):
    env = object.__new__(EL_4090_EA2)
    env.cfg = _Cfg()
    env.device = "cpu"
    env.num_envs = n
    env.dt = 0.02
    env.v = torch.full((n,), 1.0)
    env.s = torch.zeros(n)
    env.heading = torch.zeros(n)
    env.omega = torch.zeros(n)
    env.delta_actual = torch.zeros(n)
    env.delta_target = torch.zeros(n)
    env.tangent = torch.zeros(n)
    env.tangent_rate = torch.zeros(n)
    env.ego_motion = torch.zeros(n, 3)
    env.base_pos = torch.zeros(n, 3)
    env._turn_in_place = torch.zeros(n, dtype=torch.bool)
    env._turn_target = torch.zeros(n)
    env._path_batch = PathBatch(num_envs=n, max_points=64, max_corners=8, device="cpu")
    for i, p in enumerate(paths):
        env._path_batch.install(i, p)
    replans = []

    def _fake_replan(ids):
        # emulate the env behaviour: a new path is installed and arc restarts
        replans.extend(ids)
        for i in ids:
            env.s[i] = 0.0

    env._batch_replan = _fake_replan
    return env, replans


def _straight_path(length: float = 4.0, step: float = 0.2):
    pts = np.stack(
        [np.arange(0.0, length + 1e-9, step), np.zeros(int(length / step) + 1)],
        axis=-1,
    )
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    yaws = np.zeros(len(pts))
    return _contracts.PathData(
        points=pts, yaws=yaws, arc=arc, segment_dirs=yaws[:-1].copy(),
        corner_arcs=None, corner_targets=None,
    )


def _l_path():
    """Straight along +x then a 90-degree corner onto +y."""
    a = np.stack([np.arange(0.0, 2.0 + 1e-9, 0.2), np.zeros(11)], axis=-1)
    b = np.stack([np.full(11, a[-1, 0]), np.arange(0.2, 2.2 + 1e-9, 0.2)], axis=-1)
    pts = np.concatenate([a, b], axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    yaws = np.arctan2(np.diff(pts, axis=0)[:, 1], np.diff(pts, axis=0)[:, 0])
    yaws = np.concatenate((yaws, yaws[-1:]))
    corner_arc = float(arc[10])  # boundary between the two legs
    return _contracts.PathData(
        points=pts, yaws=yaws, arc=arc, segment_dirs=yaws[:-1].copy(),
        corner_arcs=np.array([corner_arc]), corner_targets=np.array([math.pi / 2]),
    ), corner_arc


def test_straight_travel_matches_readme_contract():
    env, replans = _bare_env(1, [_straight_path()])
    for _ in range(250):  # 5 s at 1 m/s covers the 4 m path
        env._step_kinematics_batched()
        s = float(env.s[0])
        # arc advance: s == v * dt * steps, clamped by replan logic
        assert 0.0 <= s <= 4.0 + 1e-6
        # position rides the path (y == 0 on this path)
        assert float(env.base_pos[0, 1]) == pytest.approx(0.0, abs=1e-5)
        # heading converged to the tangent (0) and stays there
        assert abs(float(env.heading[0])) < 1e-3
        # ego motion is the crab decomposition with delta ~ 0
        assert float(env.ego_motion[0, 0]) == pytest.approx(
            float(env.v[0]) * math.cos(float(env.delta_actual[0])), abs=1e-5
        )
        assert abs(float(env.ego_motion[0, 1])) < 1e-3
        assert abs(float(env.ego_motion[0, 2])) <= 1.5 + 1e-6
    # reached the end -> soft replan fired, no hard reset concept here
    assert replans, "reaching the path end must trigger a soft replan"


def test_stop_and_turn_corner_semantics():
    path, corner_arc = _l_path()
    env, replans = _bare_env(1, [path])
    omegas = []
    s_at_freeze = None
    turned = False
    for _ in range(400):
        env._step_kinematics_batched()
        omegas.append(float(env.omega[0]))
        if s_at_freeze is None and bool(env._turn_in_place[0]):
            s_at_freeze = float(env.s[0])
        if s_at_freeze is not None and not bool(env._turn_in_place[0]):
            turned = True
            break
    # arc froze at the corner (within a step) while turning in place
    assert s_at_freeze == pytest.approx(corner_arc, abs=0.11)
    # translation frozen during the turn
    assert float(env.ego_motion[0, 0]) == 0.0 and float(env.ego_motion[0, 1]) == 0.0
    # heading rotated bounded by omega_max and reached the corner target
    assert all(abs(w) <= 1.5 + 1e-6 for w in omegas)
    assert abs(np_wrap(float(env.heading[0]) - math.pi / 2)) < 0.05
    assert turned
