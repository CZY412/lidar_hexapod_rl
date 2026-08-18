"""Unit tests for ``range_image.py`` (README v2 §2.4).

Covers the Airy -> 450 range-image aggregation contract:

* the frozen mapping table loads with 2250 mapped rays / 450 buckets;
* each bucket is the exact minimum of its valid mapped rays;
* empty buckets are filled with ``max_range`` (5.0 m);
* rays closer than ``r_min`` or farther than ``max_range`` are ignored;
* no-hit rays (60 m) are ignored;
* normalization divides by ``max_range``.
"""

from __future__ import annotations

# Import the env package first so Isaac Gym initializes before PyTorch (the
# repo-wide convention used by all EA2 tests).
import legged_gym.envs  # noqa: F401

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as c
from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (
    load_airy_mapping_table,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.range_image import (
    aggregate_range_image,
    range_image_observation,
)

import torch

R_MIN = c.EA2_RANGE_MIN_M
R_MAX = c.EA2_RANGE_MAX_M
FAR = c.EA2_LIDAR_FAR_PLANE_M
N_RAYS = c.EA2_N_RAYS
N_BUCKETS = c.EA2_RANGE_DIM


def _make_points(e: int, n: int = N_RAYS) -> torch.Tensor:
    """Dummy point cloud; aggregation only consumes dists, so zeros are fine."""
    return torch.zeros((e, n, 3), dtype=torch.float32)


def _bucket_rays(mapping: torch.Tensor, bucket_id: int) -> torch.Tensor:
    return (mapping == bucket_id).nonzero(as_tuple=False).squeeze(1)


def test_mapping_table_loads_with_expected_counts() -> None:
    """Frozen airy_mapping.pt has 2250 selected rays and 450 unique buckets."""
    mapping = load_airy_mapping_table()

    assert isinstance(mapping, torch.Tensor)
    assert mapping.shape == (N_RAYS,)
    assert mapping.dtype == torch.int64

    mapped_mask = mapping >= 0
    assert int(mapped_mask.sum().item()) == 25 * 90  # 2250
    assert int(mapping[mapped_mask].unique().numel()) == N_BUCKETS


def test_exact_min_for_known_bucket() -> None:
    """Bucket 0 keeps the minimum distance among its mapped rays."""
    mapping = load_airy_mapping_table()
    rays = _bucket_rays(mapping, 0)
    assert rays.numel() == 5

    dists = torch.full((1, N_RAYS), FAR, dtype=torch.float32)
    values = torch.tensor([3.0, 1.5, 4.0, 2.5, 5.0], dtype=torch.float32)
    dists[0, rays] = values

    image = aggregate_range_image(
        _make_points(1), dists, mapping, max_range=R_MAX, r_min=R_MIN
    )

    assert image.shape == (1, N_BUCKETS)
    assert image[0, 0].item() == 1.5


def test_empty_bucket_fills_max_range() -> None:
    """Buckets with no valid rays are exactly max_range (5.0)."""
    mapping = load_airy_mapping_table()

    # All rays no-hit -> every bucket is empty.
    dists = torch.full((1, N_RAYS), FAR, dtype=torch.float32)
    image = aggregate_range_image(
        _make_points(1), dists, mapping, max_range=R_MAX, r_min=R_MIN
    )

    assert image.shape == (1, N_BUCKETS)
    assert torch.all(image == R_MAX)

    # With one populated bucket, a second bucket must still stay at max_range.
    dists2 = torch.full((1, N_RAYS), FAR, dtype=torch.float32)
    rays0 = _bucket_rays(mapping, 0)
    rays1 = _bucket_rays(mapping, 1)
    dists2[0, rays0] = 1.0
    dists2[0, rays1] = 2.0

    image2 = aggregate_range_image(
        _make_points(1), dists2, mapping, max_range=R_MAX, r_min=R_MIN
    )
    assert image2[0, 0].item() == 1.0
    assert image2[0, 1].item() == 2.0
    assert image2[0, 2].item() == R_MAX  # untouched empty bucket


def test_out_of_range_rays_ignored() -> None:
    """d < 0.2 and d > 5.0 are ignored; the bucket min comes only from valid rays."""
    mapping = load_airy_mapping_table()
    rays = _bucket_rays(mapping, 0)

    dists = torch.full((1, N_RAYS), FAR, dtype=torch.float32)
    # ray 0: too close, ray 1: too far, ray 2: valid.
    dists[0, rays[0]] = 0.1
    dists[0, rays[1]] = 6.0
    dists[0, rays[2]] = 2.0

    image = aggregate_range_image(
        _make_points(1), dists, mapping, max_range=R_MAX, r_min=R_MIN
    )

    assert image[0, 0].item() == 2.0


def test_all_invalid_rays_leave_bucket_empty() -> None:
    """If every ray in a bucket is outside [0.2, 5.0], the bucket is 5.0."""
    mapping = load_airy_mapping_table()
    rays = _bucket_rays(mapping, 3)

    dists = torch.full((1, N_RAYS), FAR, dtype=torch.float32)
    dists[0, rays[0]] = 0.1
    dists[0, rays[1]] = 5.5
    dists[0, rays[2]] = 0.05

    image = aggregate_range_image(
        _make_points(1), dists, mapping, max_range=R_MAX, r_min=R_MIN
    )

    assert image[0, 3].item() == R_MAX


def test_no_hit_rays_ignored() -> None:
    """60 m no-hit rays never enter a bucket's min."""
    mapping = load_airy_mapping_table()

    dists = torch.full((2, N_RAYS), FAR, dtype=torch.float32)
    # Add one valid hit in bucket 0 for env 0; env 1 remains all no-hit.
    dists[0, _bucket_rays(mapping, 0)[0]] = 1.25

    image = aggregate_range_image(
        _make_points(2), dists, mapping, max_range=R_MAX, r_min=R_MIN
    )

    assert image[0, 0].item() == 1.25
    assert torch.all(image[1] == R_MAX)


def test_normalization_divides_by_max_range() -> None:
    """range_image_observation divides by 5.0, yielding empty=1.0."""
    raw = torch.tensor(
        [[0.5, 2.5, 5.0], [1.0, 0.2, 3.75]], dtype=torch.float32
    )
    normalized = range_image_observation(raw, R_MAX)

    assert normalized.shape == raw.shape
    assert normalized.dtype == raw.dtype
    torch.testing.assert_close(
        normalized,
        torch.tensor([[0.1, 0.5, 1.0], [0.2, 0.04, 0.75]], dtype=torch.float32),
    )
