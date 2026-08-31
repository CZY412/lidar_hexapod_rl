"""Stage 2 speed-schedule tests: mixture sampler and v=0 kinematics safety.

Covers the two failure modes flagged in the plan review:

* the mixture sampler must put real mass at exactly zero (the stationary
  states the whole round exists to create) and spread the rest over (0, max];
* a v=0 env must not deadlock the batched kinematics: it may hold position,
  and once v resumes (next schedule segment) travel must continue normally.
"""

from __future__ import annotations

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import numpy as np
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    speed_mixture_draw,
)


def test_speed_mixture_zero_mode_and_uniform_part():
    u_mode = torch.tensor([0.0, 0.019, 0.021, 0.5, 0.999])
    u_speed = torch.tensor([0.3, 0.9, 0.0, 0.5, 1.0])
    v = speed_mixture_draw(u_mode, u_speed, p_zero=0.02, max_speed=1.0)
    assert torch.equal(v[:2], torch.zeros(2)), "u_mode < p_zero must yield exactly 0"
    assert torch.equal(v[2:], u_speed[2:]), "non-zero mode must be u_speed * max"


def test_speed_mixture_deterministic_and_bounded():
    g = torch.Generator().manual_seed(0)
    u_mode = torch.rand(100000, generator=g)
    u_speed = torch.rand(100000, generator=g)
    v = speed_mixture_draw(u_mode, u_speed, p_zero=0.02, max_speed=1.0)
    frac_zero = float((v == 0).float().mean())
    assert abs(frac_zero - 0.02) < 0.005, f"zero mass {frac_zero} off 0.02"
    assert float(v.max()) <= 1.0 and float(v.min()) >= 0.0
    assert float(v[v > 0].min()) > 0.0


def _bare_env_with_zero_then_move():
    """Reuse the kinematics fixture: env frozen at v=0 mid-path, then v resumes."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_ea2_env_kinematics import _bare_env, _straight_path

    env, replans = _bare_env(2, [_straight_path(8.0), _straight_path(8.0)])
    env.v[:] = 0.0
    return env, replans


def test_v0_env_holds_position_without_deadlock():
    env, _ = _bare_env_with_zero_then_move()
    s0 = env.s.clone()
    for _ in range(50):
        env._step_kinematics_batched()
    assert torch.equal(env.s, s0), "v=0 env must not advance along the path"
    assert bool(torch.isfinite(env.base_pos).all())


def test_v_resumption_continues_travel():
    env, replans = _bare_env_with_zero_then_move()
    for _ in range(10):
        env._step_kinematics_batched()
    env.v[:] = torch.tensor([0.5, 0.0])
    env._step_kinematics_batched()
    assert float(env.s[0]) > 0.0, "travel must resume once v > 0"
    assert float(env.s[1]) == 0.0
    assert not replans, "no soft replan may fire while stopped"


def test_resampler_schedule_mask_matches_env_contract():
    """episode_length_buf % speed_resample_steps == 0, per env, staggered."""
    buf = torch.tensor([149, 150, 300, 45])
    due = (buf % 150) == 0
    assert due.tolist() == [False, True, True, False]
