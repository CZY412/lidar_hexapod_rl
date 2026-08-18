"""Aggregate Airy LiDAR rays into the 450-dimensional envelope range image.

This module implements README v2 §2.4:

* each bucket keeps the minimum distance among the valid mapped rays in that
  bucket;
* valid rays satisfy ``r_min <= d <= max_range`` (defaults 0.2 m / 5.0 m);
* no-hit rays (``far_plane``) and out-of-range rays are ignored, so an empty
  bucket is filled with ``max_range``;
* the range image is normalized by ``max_range`` for the policy observation.

The input follows the frozen contract:

    points: (E, 5760, 3)   -- ray end points (used for shape compatibility)
    dists:  (E, 5760)      -- per-ray distances (source of the min aggregation)
    mapping: (5760,) int64 -- bucket id per ray, -1 for non-selected rays
    output: (E, 450)       -- row-major range image
"""

from __future__ import annotations

import torch

from ._contracts import EA2_RANGE_DIM


def aggregate_range_image(
    points: torch.Tensor,
    dists: torch.Tensor,
    mapping: torch.Tensor,
    max_range: float,
    r_min: float,
) -> torch.Tensor:
    """Aggregate per-ray distances into a fixed-size range image.

    Args:
        points: Ray end points, shape ``(E, N, 3)`` where ``N`` is the number
            of rays (typically 5760).  Kept in the signature for contract
            compatibility; aggregation uses :attr:`dists`.
        dists: Per-ray distances.  Shape ``(E, N)`` or the raw LidarSensor
            shape ``(E, 1, N, 1)``, which is flattened to ``(E, N)``.
        mapping: Mapping table of shape ``(N,)`` with dtype ``int64``.
            Each entry is the row-major bucket id in ``[0, 450)`` for selected
            rays and ``-1`` for rays outside the selected Airy channels.
        max_range: Effective maximum range.  Empty buckets are filled with
            this value; it is also the upper bound of the valid range.
        r_min: Minimum valid distance.  Rays closer than this are ignored.

    Returns:
        Float tensor of shape ``(E, 450)`` containing one minimum distance per
        bucket, with empty buckets set to :attr:`max_range`.
    """
    if dists.dim() == 4 and dists.shape[-1] == 1:
        # LidarSensor can produce (E, 1, N, 1); flatten to the contract's
        # (E, N) layout while preserving the batch dimension.
        dists = dists.reshape(dists.shape[0], -1)
    if dists.dim() != 2:
        raise ValueError(
            f"dists must be 2D (E, N) or 4D (E, 1, N, 1), got shape {tuple(dists.shape)}"
        )

    n_rays = dists.shape[1]
    if mapping.shape[0] != n_rays:
        raise ValueError(
            f"mapping has {mapping.shape[0]} entries but dists has {n_rays} rays"
        )

    device = dists.device
    dtype = dists.dtype

    mapping_dev = mapping.to(device=device)
    # scatter_reduce_ requires valid index values; non-selected -1 entries are
    # mapped to bucket 0 with src=max_range, which cannot lower any bucket's min
    # (empty buckets already start at max_range).
    bucket = mapping_dev.clamp(min=0)
    bucket_idx = bucket.unsqueeze(0).expand(dists.shape[0], -1)

    valid = (
        (dists >= r_min)
        & (dists <= max_range)
        & (mapping_dev >= 0)
    )
    src = torch.where(valid, dists, torch.full_like(dists, max_range))

    out = torch.full(
        (dists.shape[0], EA2_RANGE_DIM),
        max_range,
        dtype=dtype,
        device=device,
    )
    out.scatter_reduce_(
        1,
        bucket_idx,
        src,
        reduce="amin",
        include_self=True,
    )
    return out


def range_image_observation(
    range_image: torch.Tensor,
    max_range: float,
) -> torch.Tensor:
    """Normalize a raw range image by dividing every bucket by ``max_range``.

    Args:
        range_image: Raw aggregated range image, shape ``(..., 450)``.
        max_range: Normalization divisor (the effective max range, 5.0 m).

    Returns:
        Normalized range image of the same shape/dtype as the input.
    """
    return range_image / max_range
