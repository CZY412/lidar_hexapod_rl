"""Grid-based per-parameter envelope oracle for ``envelope_adaptive_2``.

This module computes a "theoretical target envelope" directly from the 2D
distance field, without using the LiDAR point cloud.  The envelope family is
parameterised with five independent scale factors, each derived from a group
of hexagon edge directions inspired by the legacy fan-sector grouping in
``envs/el_4090/envelope_adaptive``.

Entry points are provided:

* :func:`compute_oracle_params` returns the raw uncorrected oracle;
* :func:`compute_direct_oracle_params_with_stats` returns the Active-set
  direct per-parameter oracle used by training (with raw hard violation for
  monitoring).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    _hex_sample_violations,
    hex_body_sample_points,
    hex_collision_terms,
)

# The first 24 of the 34 hex samples are boundary samples (6 vertices + 18
# edge interpolation points).  Grouping is based on the legacy fan sectors.
GROUP_INDICES = {
    "front_width": [0, 1, 4, 5, 6, 10, 12, 16, 18, 22],
    "middle_width": [1, 4, 7, 10, 13, 15, 18, 21],
    "back_width": [2, 3, 7, 9, 13, 15, 19, 21],
    "forward_limit": [0, 5, 11, 17, 23],
    "backward_limit": [8, 14, 20],
}
PARAM_NAMES: Sequence[str] = (
    "front_width",
    "middle_width",
    "back_width",
    "forward_limit",
    "backward_limit",
)

# Boundary-only dependency groups for the direct oracle.  Unlike the legacy
# GROUP_INDICES, every boundary sample is assigned to *all* envelope
# parameters that influence its coordinates.  This still uses only the 24
# boundary samples; interior points are intentionally excluded.
#
# Hex vertex order: B(0)=fwd_width+fwd_limit, D(1)=middle_width,
# F(2)=back_width+back_limit, E(3)=back_width+back_limit,
# C(4)=middle_width, A(5)=fwd_width+fwd_limit.
# Edge interpolation indices:
#   B-D: 6,12,18; D-F: 7,13,19; F-E: 8,14,20;
#   E-C: 9,15,21; C-A: 10,16,22; A-B: 11,17,23.
_DIRECT_BOUNDARY_GROUPS = {
    "front_width": [0, 5, 6, 10, 11, 12, 16, 17, 18, 22, 23],
    "middle_width": [1, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22],
    "back_width": [2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21],
    "forward_limit": [0, 5, 6, 10, 11, 12, 16, 17, 18, 22, 23],
    "backward_limit": [2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21],
}

# Inverse mapping used by the active-set shrink: for every boundary sample
# index (0..23), which of the five envelope parameters influence it.
_SAMPLE_TO_PARAM_MASK = torch.zeros(24, len(PARAM_NAMES), dtype=torch.bool)
for _j, _idxs in enumerate(_DIRECT_BOUNDARY_GROUPS.values()):
    for _s in _idxs:
        _SAMPLE_TO_PARAM_MASK[_s, _j] = True



# The direct oracle intentionally uses only the first 24 hexagon boundary
# samples (vertices + edge interpolations).  Interior sampling is not used
# for the oracle target; it is kept for hard-collision monitoring.


_EDGE_COUNT = 24


def _physical_min_max(
    low: Tensor,
    high: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the physical minimum/maximum envelope parameter vectors.

    The backward_limit bound is reversed: a more negative value is a larger
    physical rear extent, so ``max_v`` uses ``low[4]`` and ``min_v`` uses
    ``high[4]``.
    """
    min_v = torch.stack(
        [low[0], low[1], low[2], low[3], high[4]]
    )
    max_v = torch.stack(
        [high[0], high[1], high[2], high[3], low[4]]
    )
    return min_v, max_v


def _full_edge_directions(
    low: Tensor,
    high: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return ``(edge_xy, radii)`` for the full-size hex boundary samples."""
    min_v, max_v = _physical_min_max(low, high)
    full = hex_body_sample_points(max_v.unsqueeze(0))  # (1, 34, 2)
    edge = full[0, :_EDGE_COUNT].to(device)  # (24, 2)
    radii = torch.norm(edge, dim=-1)
    return edge, radii


def _compute_raw_scales(
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    low: Tensor,
    high: Tensor,
    margin: float,
    step: float,
    max_dist: float,
    interp_crossing: bool = False,
) -> Tensor:
    """Return raw per-parameter scales, shape ``(E, 5)``.

    With ``interp_crossing=False`` (default) the first obstructed sample is
    quantised to the ray-march grid and backed off one full step, exactly as
    before.  With ``interp_crossing=True`` the ``clearance == margin``
    crossing is linearly interpolated between the last clear and the first
    obstructed sample, removing the staircase artefact (and its up to one
    step of extra conservatism) at the source.
    """
    if distance_field is None:
        raise ValueError("distance_field must be provided")
    if low.shape[-1] != 5 or high.shape[-1] != 5:
        raise ValueError("low/high must have last dimension 5")
    if step <= 0.0:
        raise ValueError("step must be positive")
    if max_dist <= 0.0:
        raise ValueError("max_dist must be positive")

    device = base_pos_xy.device
    if isinstance(distance_field, torch.Tensor):
        dist = distance_field
    else:
        dist = torch.as_tensor(distance_field, dtype=torch.float32, device=device)
    if dist.device != device:
        dist = dist.to(device)

    edge, radii = _full_edge_directions(low, high, device)

    # Rotate body-frame edge directions into world frame.
    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)
    wx = cos_h * edge[:, 0] - sin_h * edge[:, 1]
    wy = sin_h * edge[:, 0] + cos_h * edge[:, 1]
    norm = torch.hypot(wx, wy).clamp_min(1e-6)
    dir_x = wx / norm
    dir_y = wy / norm

    height, width = dist.shape
    world_min = -37.0
    res = 0.1
    # Out-of-bounds rays are treated as infinitely clear, but with a *finite*
    # sentinel so the linear interpolation denominator below cannot produce
    # inf/inf = NaN.
    clear_sentinel = max_dist + 1.0
    first_bad = torch.full(
        (base_pos_xy.shape[0], _EDGE_COUNT),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    prev_t = torch.zeros_like(first_bad)
    prev_clear = torch.full_like(first_bad, clear_sentinel)

    dist_flat = dist.reshape(-1)

    def _sample_clearance(px: Tensor, py: Tensor) -> Tensor:
        """Clearance at world (px, py).  Nearest-cell lookup by default;
        bilinear in cell centres when ``interp_crossing`` is on, so that the
        field is continuous along the ray and the interpolated crossing below
        is meaningful (with a 0.1 m grid and margin == res, nearest-cell
        clearance only takes multiples of res near a straight wall, so the
        margin contour always lands exactly on a sample and the crossing
        interpolation degenerates)."""
        if not interp_crossing:
            ix = torch.floor((px - world_min) / res).long()
            iy = torch.floor((py - world_min) / res).long()
            in_bounds = (
                (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            )
            safe_ix = ix.clamp(0, width - 1)
            safe_iy = iy.clamp(0, height - 1)
            return torch.where(
                in_bounds,
                dist_flat[safe_iy * width + safe_ix],
                torch.full_like(ix, clear_sentinel, dtype=torch.float32),
            )
        u = (px - world_min) / res - 0.5
        v = (py - world_min) / res - 0.5
        ix0 = torch.floor(u).long()
        iy0 = torch.floor(v).long()
        fx = (u - ix0.float()).clamp(0.0, 1.0)
        fy = (v - iy0.float()).clamp(0.0, 1.0)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        in_bounds = (
            (ix0 >= 0) & (iy0 >= 0) & (ix1 < width) & (iy1 < height)
        )
        cix0 = ix0.clamp(0, width - 1)
        cix1 = ix1.clamp(0, width - 1)
        ciy0 = iy0.clamp(0, height - 1)
        ciy1 = iy1.clamp(0, height - 1)
        d00 = dist_flat[ciy0 * width + cix0]
        d10 = dist_flat[ciy0 * width + cix1]
        d01 = dist_flat[ciy1 * width + cix0]
        d11 = dist_flat[ciy1 * width + cix1]
        top = d00 + (d10 - d00) * fx
        bot = d01 + (d11 - d01) * fx
        out = top + (bot - top) * fy
        return torch.where(
            in_bounds,
            out,
            torch.full_like(out, clear_sentinel),
        )

    t = step
    while t <= max_dist + 1e-9:
        px = base_pos_xy[:, 0:1] + t * dir_x
        py = base_pos_xy[:, 1:2] + t * dir_y
        clearance = _sample_clearance(px, py)
        bad = clearance < margin
        if interp_crossing:
            frac = (prev_clear - margin) / (prev_clear - clearance).clamp_min(1e-6)
            frac = frac.clamp(0.0, 1.0)
            t_cross = prev_t + frac * (t - prev_t)
            first_bad = torch.where(
                bad & torch.isinf(first_bad), t_cross, first_bad
            )
        else:
            first_bad = torch.where(bad & torch.isinf(first_bad), t, first_bad)
        prev_t = torch.full_like(prev_t, t)
        prev_clear = clearance
        t += step

    if interp_crossing:
        allowed = torch.where(
            torch.isfinite(first_bad),
            first_bad / radii.unsqueeze(0),
            torch.ones_like(first_bad),
        )
    else:
        allowed = torch.where(
            torch.isfinite(first_bad),
            ((first_bad - step) / radii.unsqueeze(0)).clamp(0.0, 1.0),
            torch.ones_like(first_bad),
        )

    scales = []
    for name in PARAM_NAMES:
        idx = GROUP_INDICES[name]
        group_allowed = allowed[:, idx]
        s = group_allowed.min(dim=-1).values
        scales.append(s.clamp(0.0, 1.0))

    return torch.stack(scales, dim=-1)  # (E, 5)


def _params_from_scales(
    scale_tensor: Tensor,
    low: Tensor,
    high: Tensor,
) -> Tensor:
    """Convert ``(E, 5)`` scales to envelope parameters."""
    min_v, max_v = _physical_min_max(low, high)
    return min_v.unsqueeze(0) + scale_tensor * (max_v - min_v).unsqueeze(0)



def _active_set_shrink_params(
    params: Tensor,
    low: Tensor,
    high: Tensor,
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    margin: float,
    soft_margin: float,
    threshold: float = 0.05,
    num_iter: int = 6,
    max_iters: int = 2,
) -> Tensor:
    """Shrink only the parameters involved in current collisions (active set).

    Instead of the old per-parameter coordinate descent, this finds each
    violating boundary sample, collects *all* envelope parameters that
    influence it, and jointly binary-searches a shrink multiplier for exactly
    that active subset.  Inactive parameters are left untouched, so this is
    not a global safety multiplier.  Several outer iterations handle newly
    exposed violations caused by shape coupling.
    """
    min_v, max_v = _physical_min_max(low, high)
    current = params.clone()
    sample_mask = _SAMPLE_TO_PARAM_MASK.to(
        device=current.device, dtype=torch.bool
    )

    denom = max_v - min_v

    for _ in range(max(1, max_iters)):
        violation = _hex_sample_violations(
            current,
            heading,
            base_pos_xy,
            distance_field,
            margin,
            soft_margin,
        )[..., :24]  # (E, 24)
        bad = violation > threshold  # (E, 24)
        if not bool(bad.any().item()):
            break

        # Parameters that influence at least one currently violating sample.
        active = (bad.unsqueeze(-1) & sample_mask.unsqueeze(0)).any(dim=1)

        scales = ((current - min_v) / denom).clamp(0.0, 1.0)

        # Per-environment binary search on a common shrink multiplier applied
        # only to active parameters.
        lo = torch.zeros(
            current.shape[0], dtype=current.dtype, device=current.device
        )
        hi = torch.ones_like(lo)

        for _ in range(max(1, num_iter)):
            mid = 0.5 * (lo + hi)
            factor = torch.where(
                active,
                mid.unsqueeze(-1),
                torch.ones_like(mid).unsqueeze(-1),
            )
            cand_scales = (scales * factor).clamp(0.0, 1.0)
            cand = min_v + cand_scales * denom

            cand_viol = _hex_sample_violations(
                cand,
                heading,
                base_pos_xy,
                distance_field,
                margin,
                soft_margin,
            )[..., :24]
            safe = (cand_viol <= threshold).all(dim=-1)
            lo = torch.where(safe, mid, lo)
            hi = torch.where(safe, hi, mid)

        factor = torch.where(
            active,
            lo.unsqueeze(-1),
            torch.ones_like(lo).unsqueeze(-1),
        )
        current = min_v + (scales * factor).clamp(0.0, 1.0) * denom

    return current


def compute_oracle_params(
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    low: Tensor,
    high: Tensor,
    margin: float = 0.10,
    step: float = 0.05,
    max_dist: float = 5.0,
    interp_crossing: bool = False,
) -> Tensor:
    """Compute raw per-parameter theoretical envelope params from the grid map.

    This is the uncorrected oracle.  Use
    :func:`compute_direct_oracle_params_with_stats` when the training reward
    should target the Active-set direct envelope.

    Args:
        heading: Body yaw, shape ``(E,)``.
        base_pos_xy: World ``(x, y)`` position, shape ``(E, 2)``.
        distance_field: 2D unsigned distance-to-obstacle field, shape
            ``(H, W)`` in metres.  Accepts ``numpy.ndarray`` or ``torch.Tensor``.
        low: Five envelope lower bounds, shape ``(5,)``.
        high: Five envelope upper bounds, shape ``(5,)``.
        margin: Safety clearance threshold used during ray marching.
        step: Ray-march step size in metres.
        max_dist: Max ray-march distance in metres.
        interp_crossing: Linearly interpolate the ``clearance == margin``
            crossing (see :func:`_compute_raw_scales`).

    Returns:
        Oracle envelope parameters, shape ``(E, 5)``.
    """
    scales = _compute_raw_scales(
        heading,
        base_pos_xy,
        distance_field,
        low,
        high,
        margin,
        step,
        max_dist,
        interp_crossing,
    )
    return _params_from_scales(scales, low, high)


def compute_direct_oracle_params_with_stats(
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    low: Tensor,
    high: Tensor,
    margin: float = 0.10,
    step: float = 0.05,
    max_dist: float = 5.0,
    safety_threshold: float = 0.05,
    num_iter: int = 6,
    max_iters: int = 2,
    collision_margin: float = 0.10,
    collision_soft_margin: float = 0.10,
    soft_dof_pos_limit: float = 0.9,
    interp_crossing: bool = False,
) -> tuple[Tensor, Tensor]:
    """Direct per-parameter oracle using active-set joint shrink.

    The raw per-parameter oracle is still computed by ray-marching.  Instead
    of a global multiplier, only the parameters involved in current boundary
    violations are jointly shrunk (active set).  The returned tuple is
    ``(params, raw_hard)``.

    Ordering contract: the expansion-side soft caps are applied BEFORE the
    active-set shrink, so the shrink stage is the last to touch the params
    and its per-sample safety verification holds for the returned values.
    (A post-shrink clamp would move samples after verification and can push
    a corner back into an obstacle cell.)

    No global safety verification/fallback is performed here; use the returned
    params only after checking the monitored ``oracle_unsafe_*`` ratios.
    """
    scales = _compute_raw_scales(
        heading,
        base_pos_xy,
        distance_field,
        low,
        high,
        margin,
        step,
        max_dist,
        interp_crossing,
    )
    raw_params = _params_from_scales(scales, low, high)

    # Align the oracle target with the policy's soft action range on the
    # *expansion* side only.  The soft bounds exist to keep actions away from
    # the hard limits; their per-parameter floors (e.g. back_width >= 0.315)
    # must NOT act as a minimum-size constraint -- when the active-set shrink
    # has to go below the soft floor to clear an obstacle, safety wins.
    # For backward_limit the expansion side is soft_low (more negative =
    # larger rear extent), so only that side is softened.
    soft_margin = (1.0 - soft_dof_pos_limit) * (high - low) / 2.0
    soft_high = high - soft_margin
    soft_low_bwd = (low + soft_margin)[4]
    clamped = [
        torch.clamp(raw_params[..., j], min=low[j].item(), max=soft_high[j].item())
        for j in range(4)
    ]
    clamped.append(
        torch.clamp(raw_params[..., 4], min=soft_low_bwd.item(), max=high[4].item())
    )
    raw_params = torch.stack(clamped, dim=-1)

    _, raw_hard = hex_collision_terms(
        raw_params,
        heading,
        base_pos_xy,
        distance_field,
        margin=collision_margin,
        soft_margin=collision_soft_margin,
    )

    direct_params = _active_set_shrink_params(
        raw_params,
        low,
        high,
        heading,
        base_pos_xy,
        distance_field,
        margin=collision_margin,
        soft_margin=collision_soft_margin,
        threshold=safety_threshold,
        num_iter=num_iter,
        max_iters=max_iters,
    )

    return direct_params, raw_hard



