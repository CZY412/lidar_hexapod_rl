"""Unit tests for ``airy_mount.py`` mapping table and body-frame coverage.

These tests cover the README v2 §2.3.1-§2.3.5 contract:

* table shape ``(5760,)`` with exactly 450 unique row-major bucket indices;
* all 25 selected azimuth channels x 90 selected elevation lines map to a
  bucket (2250 mapped rays), non-selected rays are ``-1``;
* mapping formula spot checks;
* ``self_check_mapping_table`` returns the expected diagnostics and raises on
  invalid tables;
* the saved ``airy_mapping.pt`` file equals a freshly built table;
* the boresight direction in the body frame matches the documented mount.
"""

from __future__ import annotations

import math
from pathlib import Path

# The repo's envs/__init__.py imports isaacgym, which requires torch NOT to
# have been imported yet.  Importing legged_gym modules first lets isaacgym
# initialize torch itself; all later torch imports are then safe.
from legged_gym.envs.el_4090.envelope_adaptive_2._contracts import (
    EA2_AIRY_N_ELEVATION,
    EA2_MAPPING_TABLE_FILE,
    EA2_N_COLS,
    EA2_N_ROWS,
    EA2_N_RAYS,
    EA2_RANGE_DIM,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (
    body_frame_ray_directions,
    build_airy_mapping_table,
    load_airy_mapping_table,
    save_airy_mapping_table,
    self_check_mapping_table,
)

import pytest
import torch


def test_mapping_table_shape_and_bucket_count() -> None:
    """The table covers all rays and maps to exactly 450 unique buckets."""
    table = build_airy_mapping_table()

    assert isinstance(table, torch.Tensor)
    assert table.shape == (EA2_N_RAYS,)
    assert table.dtype == torch.int64

    mapped_mask = table >= 0
    n_mapped_rays = int(mapped_mask.sum().item())
    n_unique_buckets = int(table[mapped_mask].unique().numel())

    # README §2.3.3/§2.3.4: 25 az * 90 el = 2250 rays are selected and each
    # maps to one of 18*25=450 row-major buckets.  The user-facing count of
    # "450" refers to the number of range-image buckets, not the number of
    # non-(-1) ray entries.
    assert n_mapped_rays == 25 * 90
    assert n_unique_buckets == EA2_RANGE_DIM
    assert int((table == -1).sum().item()) == EA2_N_RAYS - n_mapped_rays


def test_mapping_formula_spot_checks() -> None:
    """Flatten-index and row-major bucket formulas are exact."""
    table = build_airy_mapping_table()

    # Selected first/last rays in the first/last buckets.
    assert table[18 * EA2_AIRY_N_ELEVATION + 6] == 0
    assert table[18 * EA2_AIRY_N_ELEVATION + 10] == 0
    assert table[18 * EA2_AIRY_N_ELEVATION + 11] == 25
    assert table[30 * EA2_AIRY_N_ELEVATION + 6] == 12
    assert table[30 * EA2_AIRY_N_ELEVATION + 95] == (EA2_N_ROWS - 1) * EA2_N_COLS + 12
    assert table[42 * EA2_AIRY_N_ELEVATION + 95] == EA2_RANGE_DIM - 1

    # Non-selected rays stay -1.
    assert table[17 * EA2_AIRY_N_ELEVATION + 6] == -1
    assert table[43 * EA2_AIRY_N_ELEVATION + 6] == -1
    assert table[18 * EA2_AIRY_N_ELEVATION + 5] == -1

    # Full vectorized formula check for every mapped ray.
    mapped_indices = torch.nonzero(table >= 0, as_tuple=False).squeeze(1)
    az = mapped_indices // EA2_AIRY_N_ELEVATION
    el = mapped_indices % EA2_AIRY_N_ELEVATION
    expected = (el - 6) // 5 * EA2_N_COLS + (az - 18)
    assert torch.equal(table[mapped_indices], expected)


def test_self_check_returns_assertions() -> None:
    """Self-check returns diagnostics and rejects invalid tables."""
    table = build_airy_mapping_table()
    result = self_check_mapping_table(table)

    assert isinstance(result, dict)
    assert result["n_mapped"] == 25 * 90
    assert result["n_unique_buckets"] == EA2_RANGE_DIM
    assert result["range_dim"] == EA2_RANGE_DIM
    assert result["all_body_x_positive"] is True
    assert result["mapping_formula_consistent"] is True

    az_min, az_max = result["azimuth_deg_range"]
    el_min, el_max = result["elevation_deg_range"]
    assert -80.0 <= az_min <= az_max <= 80.0
    assert -25.0 <= el_min <= el_max <= 70.0

    # A corrupted table must trip the assertion.
    bad = table.clone()
    bad[18 * EA2_AIRY_N_ELEVATION + 6] = -1
    with pytest.raises(AssertionError):
        self_check_mapping_table(bad)

    with pytest.raises(AssertionError):
        self_check_mapping_table(torch.zeros(10, dtype=torch.int64))


def test_saved_file_loads_and_equals_built_table() -> None:
    """Default saved artifact round-trips and matches a fresh build."""
    table = build_airy_mapping_table()
    saved_path = save_airy_mapping_table(table)

    assert Path(saved_path).exists()
    loaded = load_airy_mapping_table(saved_path)
    assert isinstance(loaded, torch.Tensor)
    assert torch.equal(loaded, table)

    # Also verify the frozen default path is the one referenced by contracts.
    assert Path(EA2_MAPPING_TABLE_FILE).exists()
    assert torch.equal(load_airy_mapping_table(), table)


def test_boresight_direction_check() -> None:
    """Sensor +z boresight maps to body (0.939, 0, -0.343) per README."""
    dirs = body_frame_ray_directions()

    assert dirs.shape == (EA2_N_RAYS, 3)
    assert dirs.dtype == torch.float32

    # el=95 (phi=90 deg) is the sensor +z direction; all az share the same
    # direction there, so i=95 is a valid representative.
    boresight = dirs[95]
    expected = torch.tensor(
        [0.9393727, 0.0, -0.3428978], dtype=torch.float32
    )
    assert torch.allclose(boresight, expected, atol=1e-5, rtol=1e-5)

    # Direct trigonometric check using the documented offset rpy.  A +z
    # sensor ray rotated about the body y-axis by pitch maps to
    # (sin(pitch), 0, cos(pitch)).
    pitch = math.pi / 2.0 + 0.35
    direct_expected = torch.tensor(
        [math.sin(pitch), 0.0, math.cos(pitch)], dtype=torch.float32
    )
    assert torch.allclose(boresight, direct_expected, atol=1e-6, rtol=1e-6)
