"""Tests for the fixed 187-channel range image in ``range_image.py``."""

from __future__ import annotations

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as c
from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (
    load_selected_channels,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.range_image import (
    build_selected_range_image,
    extract_selected_range_image,
    range_image_observation,
)

import pytest
import torch

R_MAX = c.EA2_RANGE_MAX_M
FAR = c.EA2_LIDAR_FAR_PLANE_M
N = c.EA2_RANGE_DIM


def test_build_selected_range_image_keeps_valid_hits() -> None:
    dists = torch.tensor(
        [[0.5, 1.2, FAR, 2.5, 10.0] + [3.0] * (N - 5)],
        dtype=torch.float32,
    )
    out = build_selected_range_image(dists, R_MAX)
    assert out.shape == (1, N)
    assert out[0, 0].item() == pytest.approx(0.5)
    assert out[0, 1].item() == pytest.approx(1.2)
    assert out[0, 2].item() == pytest.approx(R_MAX)  # no-hit
    assert out[0, 3].item() == pytest.approx(2.5)
    assert out[0, 4].item() == pytest.approx(R_MAX)  # beyond max_range


def test_build_selected_range_image_accepts_4d_input() -> None:
    dists = torch.full((2, 1, N, 1), FAR, dtype=torch.float32)
    dists[0, 0, 0, 0] = 1.0
    out = build_selected_range_image(dists, R_MAX)
    assert out.shape == (2, N)
    assert out[0, 0].item() == pytest.approx(1.0)
    assert out[0, 1].item() == pytest.approx(R_MAX)


def test_extract_selected_range_image() -> None:
    data = load_selected_channels()
    indices = data["ray_indices"]

    full_dists = torch.full((2, c.EA2_FULL_N_RAYS), FAR, dtype=torch.float32)
    # Put a valid hit on the first selected ray of env 0.
    full_dists[0, indices[0]] = 0.8
    out = extract_selected_range_image(full_dists, indices, R_MAX)
    assert out.shape == (2, N)
    assert out[0, 0].item() == pytest.approx(0.8)
    assert out[1, 0].item() == pytest.approx(R_MAX)


def test_normalization_divides_by_max_range() -> None:
    raw = torch.tensor([[0.0, R_MAX / 2.0, R_MAX]], dtype=torch.float32)
    norm = range_image_observation(raw, R_MAX)
    torch.testing.assert_close(
        norm,
        torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32),
    )
