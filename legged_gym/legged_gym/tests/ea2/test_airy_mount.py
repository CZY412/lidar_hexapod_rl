"""Tests for the fixed 187-channel Airy selection in ``airy_mount.py``."""

from __future__ import annotations

from pathlib import Path

# The repo's envs/__init__.py imports isaacgym, which requires torch NOT to
# have been imported yet.  Importing the EA2 package first lets isaacgym
# initialize torch itself; all later torch imports are then safe.
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as c
from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (
    body_frame_ray_directions,
    generate_full_airy_directions,
    load_selected_channels,
    save_selected_channels,
    select_ground_grid_channels,
    self_check_selected_channels,
)

import pytest
import torch


def test_full_airy_directions_shape() -> None:
    """Full Airy pattern is 900 x 96 = 86,400 rays."""
    dirs = generate_full_airy_directions()
    assert dirs.shape == (c.EA2_FULL_N_RAYS, 3)
    assert dirs.dtype == torch.float64
    # Unit directions.
    norms = torch.norm(dirs, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_body_frame_ray_directions_shape() -> None:
    dirs = body_frame_ray_directions()
    assert dirs.shape == (c.EA2_FULL_N_RAYS, 3)
    assert dirs.dtype == torch.float32


def test_selected_channels_count_unique_and_range() -> None:
    """187 channels, unique, and max_range rounded up to 0.1 m."""
    data = select_ground_grid_channels()

    assert data["ray_indices"].shape == (c.EA2_RANGE_DIM,)
    assert data["ray_directions"].shape == (c.EA2_RANGE_DIM, 3)
    assert data["cell_centers"].shape == (c.EA2_RANGE_DIM, 2)
    assert data["ground_hits"].shape == (c.EA2_RANGE_DIM, 2)
    assert data["slant_ranges"].shape == (c.EA2_RANGE_DIM,)

    assert len(torch.unique(data["ray_indices"])) == c.EA2_RANGE_DIM
    assert data["max_range"] == pytest.approx(
        round(float(data["slant_ranges"].max().item()) * 10.0) / 10.0,
        abs=1e-9,
    )


def test_selected_channels_self_check() -> None:
    data = select_ground_grid_channels()
    result = self_check_selected_channels(data)
    assert result["n_channels"] == c.EA2_RANGE_DIM
    assert result["unique"] is True
    assert result["all_hits_in_region"] is True
    assert result["max_range"] == c.EA2_RANGE_MAX_M


def test_selected_channels_save_load_roundtrip() -> None:
    data = select_ground_grid_channels()
    path = save_selected_channels(data, "/tmp/ea2_selected_channels_test.pt")
    loaded = load_selected_channels(path)
    assert torch.equal(loaded["ray_indices"], data["ray_indices"])
    assert torch.allclose(loaded["ray_directions"], data["ray_directions"])
    assert loaded["max_range"] == data["max_range"]
    Path(path).unlink()


def test_saved_default_file_exists_and_valid() -> None:
    """The frozen default file must exist and pass self-check."""
    assert c.EA2_SELECTED_CHANNELS_FILE.exists()
    data = load_selected_channels()
    self_check_selected_channels(data)
