"""Tests for LidarSensor.apply_noise (envelope_adaptive_2 shared util patch)."""

from types import SimpleNamespace
from typing import Tuple

# Import the env package first to fully initialize legged_gym.utils without
# hitting its circular import (utils -> task_registry -> envs -> utils).
# This also loads Isaac Gym before PyTorch, which Isaac Gym requires.
import legged_gym.envs  # noqa: F401

from legged_gym.utils.LidarSensor.lidar_sensor import LidarSensor

import torch

FAR_PLANE = 60.0
SHAPE = (2, 1, 64, 1)
PIX_SHAPE = SHAPE + (3,)


def _make_dummy(
    enable_sensor_noise: bool,
    pixel_std_dev_multiplier: float = 0.02,
    pixel_dropout_prob: float = 0.02,
    far_plane: float = FAR_PLANE,
) -> SimpleNamespace:
    """Create a lightweight stand-in exposing the sensor fields apply_noise reads."""
    sensor_cfg = SimpleNamespace(
        enable_sensor_noise=enable_sensor_noise,
        pixel_std_dev_multiplier=pixel_std_dev_multiplier,
        pixel_dropout_prob=pixel_dropout_prob,
    )
    return SimpleNamespace(sensor_cfg=sensor_cfg, far_plane=far_plane)


def _make_rays(seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return original directions, pixels (dir * dist) and dists for a dummy scan."""
    torch.manual_seed(seed)
    directions = torch.randn(PIX_SHAPE, dtype=torch.float32)
    directions = directions / torch.norm(
        directions, dim=-1, keepdim=True
    ).clamp_min(1e-6)

    dists = torch.empty(SHAPE, dtype=torch.float32)
    dists[..., :32, :] = torch.linspace(0.5, 4.9, 32, dtype=torch.float32).reshape(1, 1, 32, 1)
    dists[..., 32:, :] = FAR_PLANE

    pixels = directions * dists.unsqueeze(-1)
    return directions, pixels, dists


def test_apply_noise_disabled_returns_identical_tensors() -> None:
    """When enable_sensor_noise is False, inputs are returned unchanged."""
    dummy = _make_dummy(enable_sensor_noise=False)
    _, pixels, dists = _make_rays()
    pixels_before = pixels.clone()
    dists_before = dists.clone()

    out_pixels, out_dists = LidarSensor.apply_noise(dummy, pixels, dists)

    assert out_pixels is pixels
    assert out_dists is dists
    assert torch.equal(out_pixels, pixels_before)
    assert torch.equal(out_dists, dists_before)


def test_apply_noise_enabled_preserves_directions_and_shapes() -> None:
    """Multiplicative range noise must keep each ray on its original direction."""
    dummy = _make_dummy(enable_sensor_noise=True, pixel_std_dev_multiplier=0.02, pixel_dropout_prob=0.02)
    directions, pixels, dists = _make_rays(seed=123)

    out_pixels, out_dists = LidarSensor.apply_noise(dummy, pixels, dists)

    assert out_pixels.shape == PIX_SHAPE
    assert out_dists.shape == SHAPE
    assert out_pixels.dtype == torch.float32
    assert out_dists.dtype == torch.float32

    out_norm = torch.norm(out_pixels, dim=-1, keepdim=True).clamp_min(1e-6)
    out_directions = out_pixels / out_norm
    assert torch.allclose(out_directions, directions, atol=1e-5, rtol=1e-5)


def test_apply_noise_no_hit_values_stay_out_of_effective_bucket_range() -> None:
    """No-hit 60m rays with 2% std must never enter the [0.2, 5.0] bucket range."""
    dummy = _make_dummy(enable_sensor_noise=True, pixel_std_dev_multiplier=0.02, pixel_dropout_prob=0.02)
    _, pixels, dists = _make_rays(seed=7)
    no_hit_mask = dists == FAR_PLANE

    _, out_dists = LidarSensor.apply_noise(dummy, pixels, dists.clone())

    no_hit_after = out_dists[no_hit_mask]
    assert not ((no_hit_after >= 0.2) & (no_hit_after <= 5.0)).any()


def test_apply_noise_dropout_sets_dist_to_far_plane() -> None:
    """Dropout must reset the range to far_plane and rebuild the point from direction."""
    dummy = _make_dummy(enable_sensor_noise=True, pixel_std_dev_multiplier=0.0, pixel_dropout_prob=1.0)
    directions, pixels, dists = _make_rays(seed=42)

    out_pixels, out_dists = LidarSensor.apply_noise(dummy, pixels, dists)

    assert torch.all(out_dists == FAR_PLANE)
    assert torch.allclose(out_pixels, directions * FAR_PLANE, atol=1e-5, rtol=1e-5)
