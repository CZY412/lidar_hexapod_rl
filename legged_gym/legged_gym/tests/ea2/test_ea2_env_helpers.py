"""Tests for the pure helpers in ``el_4090_ea2_env.py``.

These tests deliberately do not create an Isaac Gym simulation: they exercise
the module-level functions used by the environment for height/wobble, heading
control, rewards, observation assembly, empty-frame reset and path/sway checks.
"""

from __future__ import annotations

import math

# Isaac Gym requires that torch is imported only after isaacgym modules have
# been initialized.  Importing the EA2 env module first triggers the
# ``legged_gym.envs`` package (and therefore isaacgym) before torch is used.
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    action_rate_term,
    assemble_observation,
    collision_ratio,
    ego_motion,
    empty_range_image,
    heading_update,
    height_step,
    interpolate_path,
    map_actions_to_params,
    potential_reward,
    refresh_range_image_from_scan,
    point_cloud_debug_masks,
    sway_position_acceptable,
    sway_update,
    wrap_to_pi,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.path_planner import PathData

import numpy as np
import pytest
import torch

# Default envelope ranges from the frozen spider_envelop config (first 5).
_LOW = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9], dtype=torch.float32)
_HIGH = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6], dtype=torch.float32)


def _straight_path(length: float = 8.0, step: float = 0.2) -> PathData:
    """A straight +x path with tangent yaw 0."""
    xs = np.arange(0.0, length + 1e-9, step)
    points = np.stack([xs, np.zeros_like(xs)], axis=-1).astype(np.float64)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    yaws = np.zeros_like(xs)
    return PathData(points=points, yaws=yaws, arc=arc)


def test_point_cloud_debug_masks_match_readme_27() -> None:
    """Red/green classification follows README 2.7."""
    mapping = torch.full((8,), -1, dtype=torch.int64)
    mapping[1] = 0
    mapping[3] = 1
    dists = torch.tensor(
        [
            [0.1, 2.0, 4.9, 5.1, 60.0, 0.5, 5.0, 60.0],
            [0.1, 2.0, 4.9, 5.1, 60.0, 0.5, 5.0, 60.0],
        ],
        dtype=torch.float32,
    )
    red, green = point_cloud_debug_masks(
        dists,
        mapping,
        far_plane=60.0,
        max_range=5.0,
        r_min=0.2,
    )
    # red: mapped channels inside [0.2, 5.0] and real hits
    assert red[0].tolist() == [False, True, False, False, False, False, False, False]
    # green: every real hit that is not red (below/above agg range included)
    assert green[0].tolist() == [True, False, True, True, False, True, True, False]
    # no-hit 60 m is neither red nor green
    assert not red[:, 4].any() and not green[:, 4].any()


def test_height_step_converges_to_target_and_stays_in_bounds() -> None:
    """Height low-pass filter converges to target and the result is clipped."""
    h_filter = torch.tensor([0.53], dtype=torch.float32)
    h_target = torch.tensor([0.64], dtype=torch.float32)
    wobble = torch.zeros(1, dtype=torch.float32)
    for _ in range(5000):
        h_filter, wobble, h = height_step(
            h_filter,
            h_target,
            wobble,
            torch.zeros(1, dtype=torch.float32),
            min_m=0.53,
            max_m=0.64,
            dt=0.02,
            tau_s=0.8,
            wobble_fc_hz=1.0,
            wobble_amp=0.01,
        )
    assert h_filter.item() == pytest.approx(0.64, abs=1e-3)
    assert h.item() == pytest.approx(0.64, abs=1e-3)
    assert 0.53 <= h.item() <= 0.64 + 1e-6


def test_height_step_wobble_state_is_ar_with_unit_noise_scale() -> None:
    """A unit noise impulse moves the AR state by sqrt(1-a^2) ~= a fraction."""
    state = torch.zeros(1, dtype=torch.float32)
    state, offset = sway_update(
        state, amp=0.02, dt=0.02, fc_hz=1.0,
        noise=torch.ones(1, dtype=torch.float32),
    )
    a = np.exp(-2.0 * np.pi * 1.0 * 0.02)
    assert state.item() == pytest.approx(np.sqrt(1.0 - a * a), abs=1e-6)
    assert offset.item() == pytest.approx(0.02 * state.item(), abs=1e-6)


def test_sway_update_decays_to_zero_without_noise() -> None:
    """Without driving noise the low-pass sway state decays geometrically."""
    state = torch.tensor([1.0], dtype=torch.float32)
    for _ in range(2000):
        state, _ = sway_update(
            state, amp=0.1, dt=0.02, fc_hz=1.0,
            noise=torch.zeros(1, dtype=torch.float32),
        )
    assert state.item() == pytest.approx(0.0, abs=1e-6)


def test_heading_update_converges_to_delta_target_on_straight_path() -> None:
    """Tangent-relative heading controller drives relative yaw to target."""
    heading = torch.tensor([0.2], dtype=torch.float32)
    tangent = torch.tensor([0.0], dtype=torch.float32)
    delta_target = torch.tensor([0.1], dtype=torch.float32)
    for _ in range(2000):
        heading, omega, delta_actual = heading_update(
            heading,
            tangent,
            torch.tensor([0.0], dtype=torch.float32),
            delta_target,
            dt=0.02,
            k_p=5.0,
            omega_max=1.5,
        )
        assert abs(omega.item()) <= 1.5 + 1e-6
    assert heading.item() == pytest.approx(0.1, abs=1e-3)
    assert delta_actual.item() == pytest.approx(0.1, abs=1e-3)


def test_ego_motion_crab_decomposition() -> None:
    """Ego motion is the body-frame decomposition of v around delta."""
    v = torch.tensor([1.0], dtype=torch.float32)
    heading = torch.tensor([0.2], dtype=torch.float32)
    tangent = torch.tensor([0.0], dtype=torch.float32)
    omega = torch.tensor([0.3], dtype=torch.float32)
    vx, vy, out_omega = ego_motion(v, heading, tangent, omega)
    assert vx.item() == pytest.approx(np.cos(0.2), abs=1e-6)
    assert vy.item() == pytest.approx(np.sin(0.2), abs=1e-6)
    assert out_omega.item() == pytest.approx(0.3, abs=1e-6)


def test_map_actions_to_params_sigmoid_affine() -> None:
    """Raw actions are mapped into the envelope parameter bounds."""
    actions = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    mapped = map_actions_to_params(actions, _LOW, _HIGH)
    expected_mid = (_LOW + _HIGH) / 2.0
    torch.testing.assert_close(mapped[0], expected_mid, atol=1e-5, rtol=1e-5)

    large = torch.tensor([[50.0, 50.0, 50.0, 50.0, 50.0]], dtype=torch.float32)
    mapped_large = map_actions_to_params(large, _LOW, _HIGH)
    torch.testing.assert_close(mapped_large[0], _HIGH, atol=1e-4, rtol=1e-4)


def test_potential_reward_max_min_and_backward_direction() -> None:
    """Potential reward uses physical direction (larger backward -> higher)."""
    max_params = torch.tensor([[0.6, 0.7, 0.6, 0.9, -0.9]], dtype=torch.float32)
    min_params = torch.tensor([[0.3, 0.3, 0.3, 0.6, -0.6]], dtype=torch.float32)
    assert potential_reward(max_params, _LOW, _HIGH).item() == pytest.approx(1.0, abs=1e-5)
    assert potential_reward(min_params, _LOW, _HIGH).item() == pytest.approx(0.0, abs=1e-5)

    # A physically larger rear extent (-0.9) must score higher than a
    # physically smaller one (-0.6).
    p_large = potential_reward(
        torch.tensor([[0.4, 0.5, 0.4, 0.7, -0.9]], dtype=torch.float32),
        _LOW,
        _HIGH,
    )
    p_small = potential_reward(
        torch.tensor([[0.4, 0.5, 0.4, 0.7, -0.6]], dtype=torch.float32),
        _LOW,
        _HIGH,
    )
    assert p_large.item() > p_small.item()


def test_collision_ratio_synthetic_occupancy() -> None:
    """Collision ratio counts covered occupied cells / covered cells."""
    occupancy = np.zeros(ea2c.EA2_GRID_SHAPE, dtype=np.uint8)
    occupancy[60, 60] = 1
    occupancy[60, 61] = 1
    hex_vertices = torch.tensor(
        [
            [0.20, 0.20],  # B
            [0.10, 0.20],  # D
            [0.00, 0.20],  # F
            [0.00, 0.00],  # E
            [0.10, 0.00],  # C
            [0.20, 0.00],  # A
        ],
        dtype=torch.float32,
    )
    ratio = collision_ratio(hex_vertices, occupancy, margin=0.0)
    assert ratio.dim() == 0
    assert ratio.item() == pytest.approx(0.5, abs=1e-5)

    empty = np.zeros(ea2c.EA2_GRID_SHAPE, dtype=np.uint8)
    assert collision_ratio(hex_vertices, empty, margin=0.0).item() == pytest.approx(0.0, abs=1e-6)


def test_action_rate_term() -> None:
    """Action-rate is mean squared mapped-parameter change."""
    a = torch.zeros(3, 5, dtype=torch.float32)
    b = torch.zeros(3, 5, dtype=torch.float32)
    torch.testing.assert_close(action_rate_term(a, b), torch.zeros(3))

    a2 = torch.zeros(2, 5, dtype=torch.float32)
    b2 = torch.ones(2, 5, dtype=torch.float32)
    torch.testing.assert_close(action_rate_term(a2, b2), torch.ones(2))


def test_assemble_observation_shape_and_normalization() -> None:
    """Obs = normalized range (450) + normalized ego (3)."""
    range_image = torch.full((2, 450), 5.0, dtype=torch.float32)
    range_image[0, 0] = 2.5
    ego = torch.tensor(
        [[0.75, 0.5, 0.75], [0.0, 0.0, 0.0]], dtype=torch.float32
    )
    obs = assemble_observation(range_image, ego)
    assert obs.shape == (2, 453)
    assert obs[0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert obs[0, 1:450].abs().max().item() == pytest.approx(1.0, abs=1e-6)
    assert obs[0, 450:].tolist() == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)


def test_empty_range_image() -> None:
    """Empty-frame reset returns a full-max_range buffer."""
    img = empty_range_image(4, max_range=5.0)
    assert img.shape == (4, 450)
    assert bool((img == 5.0).all())


def test_refresh_range_image_from_scan_gives_stale_envs_fresh_frame() -> None:
    """A global scan must replace the reset empty frame with the fresh scan.

    README 2.4: reset-before-scan envs keep the all-max_range empty frame only
    until the next global scan; at that scan they receive the fresh aggregate.
    """
    range_image = empty_range_image(3, max_range=5.0)
    fresh = torch.full((3, ea2c.EA2_RANGE_DIM), 2.5, dtype=torch.float32)
    fresh[0, 0] = 0.5
    fresh[2, 0] = 1.25
    stale = torch.tensor([True, True, False], dtype=torch.bool)

    out = refresh_range_image_from_scan(range_image, fresh, stale)

    assert out is range_image
    torch.testing.assert_close(range_image, fresh)
    assert not bool(stale.any()), "stale markers must clear after a global scan"
    assert range_image[0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert range_image[1, 0].item() == pytest.approx(2.5, abs=1e-6)
    assert range_image[2, 0].item() == pytest.approx(1.25, abs=1e-6)


def test_interpolate_path_straight() -> None:
    """Interpolate path returns positions, tangent and curvature along a line."""
    path = _straight_path()
    xy, tangent, tangent_rate = interpolate_path(path, 2.5)
    assert xy[0] == pytest.approx(2.5, abs=1e-6)
    assert xy[1] == pytest.approx(0.0, abs=1e-6)
    assert tangent == pytest.approx(0.0, abs=1e-6)
    assert tangent_rate == pytest.approx(0.0, abs=1e-6)


def test_sway_position_acceptable_rejects_blocked_cells() -> None:
    """Sway acceptance rejects a candidate in an inflated occupied cell."""
    inflated = np.zeros(ea2c.EA2_GRID_SHAPE, dtype=np.uint8)
    # Mark a single cell at world (2.0, 2.0) blocked.
    ix = int(np.floor((2.0 - ea2c.EA2_WORLD_MIN_XY) / ea2c.EA2_RESOLUTION_M))
    iy = int(np.floor((2.0 - ea2c.EA2_WORLD_MIN_XY) / ea2c.EA2_RESOLUTION_M))
    inflated[iy, ix] = 1
    assert sway_position_acceptable((1.0, 1.0), (1.0, 1.1), inflated)
    assert not sway_position_acceptable((1.0, 1.0), (2.0, 2.0), inflated)


def test_wrap_to_pi() -> None:
    """wrap_to_pi returns values in [-pi, pi)."""
    a = torch.tensor([3.5, -3.5, 0.0, 6.0], dtype=torch.float32)
    wrapped = wrap_to_pi(a)
    assert bool((wrapped >= -math.pi - 1e-6).all())
    assert bool((wrapped < math.pi - 1e-6).all())
    assert wrapped[0].item() == pytest.approx(3.5 - 2.0 * math.pi, abs=1e-6)
