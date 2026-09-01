"""Bit-exact reference implementations of the envelope-oracle marches.

WHY THIS FILE EXISTS
--------------------
The optimisations in ``envelope_oracle`` are only safe if they are *bitwise*
identical to the implementation they replace.  Once the production code is
changed the previous implementation no longer exists anywhere, so a test
cannot compare "old vs new" by importing it.

This module therefore freezes a verbatim copy of the pre-optimisation
implementations.  Tests compare the production code against these frozen
copies and assert ``torch.equal``.

DO NOT "UPDATE" THESE COPIES
----------------------------
If a change in ``envelope_oracle`` makes a test here fail, that is the test
doing its job: either the change is not bitwise-equivalent (and must be
reworked), or it is an *intentional* semantic change -- in which case these
reference functions must be retired together with the equivalence tests, in
a commit whose message states the intentional behaviour change.

Never edit these functions to make a failing test pass.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

# ``_EDGE_COUNT`` mirrors the private constant in envelope_oracle.  It is
# duplicated (rather than imported) on purpose: this module must remain
# byte-stable even if the production module's internals are reorganised.
_EDGE_COUNT = 24


def axis_march_crossing(
    sample,
    start_x: Tensor,
    start_y: Tensor,
    dir_x: Tensor,
    dir_y: Tensor,
    margin: float,
    step: float,
    max_dist: float,
    interp_crossing: bool,
) -> Tensor:
    """Frozen copy of ``envelope_oracle._axis_march_crossing``.

    NOTE: reproduces the original semantics exactly, including the explicit
    ``t = 0`` sample that runs *before* the loop and the hard-coded ``(E, 1)``
    shape of ``first_bad`` (which degenerates when ``step > max_dist``).
    """
    first_bad = torch.full((start_x.shape[0], 1), float("inf"),
                           dtype=torch.float32, device=start_x.device)
    prev_t = torch.zeros_like(first_bad)
    prev_clear = sample(start_x, start_y)
    # The t=0 check runs BEFORE the loop, with the t=0 clearance.
    bad0 = prev_clear < margin
    first_bad = torch.where(bad0, torch.zeros_like(first_bad), first_bad)
    t = step
    while t <= max_dist + 1e-9:
        clearance = sample(start_x + t * dir_x, start_y + t * dir_y)
        bad = clearance < margin
        if interp_crossing:
            frac = (prev_clear - margin) / (prev_clear - clearance).clamp_min(1e-6)
            frac = frac.clamp(0.0, 1.0)
            t_cross = prev_t + frac * (t - prev_t)
            first_bad = torch.where(bad & torch.isinf(first_bad), t_cross, first_bad)
        else:
            first_bad = torch.where(bad & torch.isinf(first_bad), t, first_bad)
        prev_t = torch.full_like(prev_t, t)
        prev_clear = clearance
        t += step
    return first_bad


def raw_scales_coupled(
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    low: Tensor,
    high: Tensor,
    margin: float,
    step: float,
    max_dist: float,
    interp_crossing: bool,
) -> Tensor:
    """Frozen copy of the ``"coupled"`` branch of ``_compute_raw_scales``.

    Production runs ``"axis"``, but several tests and validation scripts still
    exercise ``"coupled"`` and one test compares the two modes, so this path is
    frozen too: it must stay bit-exact unless a commit intentionally changes
    coupled semantics.

    NOTE: the 24-direction march uses ``_full_edge_directions``, i.e. the
    sample points of ``max_v = _physical_min_max(low, high)[1]``.  An earlier
    version of this reference used ``stack([low, high])`` instead, which is
    wrong and would have silently validated a different march.
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2 import envelope_oracle as eo
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
        hex_body_sample_points,
    )

    device = base_pos_xy.device
    if isinstance(distance_field, torch.Tensor):
        dist = distance_field
    else:
        dist = torch.as_tensor(distance_field, dtype=torch.float32, device=device)
    if dist.device != device:
        dist = dist.to(device)
    sample = eo._make_field_sampler(dist, interp_crossing, max_dist)

    min_v, max_v = eo._physical_min_max(low, high)
    full = hex_body_sample_points(max_v.unsqueeze(0))
    edge = full[0, :_EDGE_COUNT].to(device)
    radii = torch.norm(edge, dim=-1)

    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)
    wx = cos_h * edge[:, 0] - sin_h * edge[:, 1]
    wy = sin_h * edge[:, 0] + cos_h * edge[:, 1]
    norm = torch.hypot(wx, wy).clamp_min(1e-6)
    dir_x, dir_y = wx / norm, wy / norm

    first_bad = torch.full((base_pos_xy.shape[0], _EDGE_COUNT), float("inf"),
                           dtype=torch.float32, device=device)
    prev_t = torch.zeros_like(first_bad)
    prev_clear = torch.full_like(first_bad, max_dist + 1.0)

    t = step
    while t <= max_dist + 1e-9:
        px = base_pos_xy[:, 0:1] + t * dir_x
        py = base_pos_xy[:, 1:2] + t * dir_y
        clearance = sample(px, py)
        bad = clearance < margin
        if interp_crossing:
            frac = (prev_clear - margin) / (prev_clear - clearance).clamp_min(1e-6)
            frac = frac.clamp(0.0, 1.0)
            t_cross = prev_t + frac * (t - prev_t)
            first_bad = torch.where(bad & torch.isinf(first_bad), t_cross, first_bad)
        else:
            first_bad = torch.where(bad & torch.isinf(first_bad), t, first_bad)
        prev_t = torch.full_like(prev_t, t)
        prev_clear = clearance
        t += step

    if interp_crossing:
        allowed = torch.where(torch.isfinite(first_bad),
                              first_bad / radii.unsqueeze(0),
                              torch.ones_like(first_bad))
    else:
        allowed = torch.where(
            torch.isfinite(first_bad),
            ((first_bad - step) / radii.unsqueeze(0)).clamp(0.0, 1.0),
            torch.ones_like(first_bad))

    scales = []
    for name in eo.PARAM_NAMES:
        idx = eo.GROUP_INDICES[name]
        scales.append(allowed[:, idx].min(dim=-1).values.clamp(0.0, 1.0))
    return torch.stack(scales, dim=-1)


def raw_scales_axis(
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    low: Tensor,
    high: Tensor,
    margin: float,
    step: float,
    max_dist: float,
    interp_crossing: bool,
) -> Tensor:
    """Frozen copy of the **axis branch** of ``_compute_raw_scales``.

    Only the axis branch is reproduced here (not the 24-direction main march),
    because in axis mode the main march result is returned unused -- it is dead
    code.  See ``test_oracle_march_equivalence.py``.
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2 import envelope_oracle as eo

    if isinstance(distance_field, torch.Tensor):
        dist = distance_field
    else:
        dist = torch.as_tensor(distance_field, dtype=torch.float32,
                               device=base_pos_xy.device)
    if dist.device != base_pos_xy.device:
        dist = dist.to(base_pos_xy.device)

    sample = eo._make_field_sampler(dist, interp_crossing, max_dist)

    fwd_min = float(low[3])
    fwd_span = float(high[3] - low[3])
    lat_min = float(low[0])
    lat_span = float(high[0] - low[0])
    bx = base_pos_xy[:, 0:1]
    by = base_pos_xy[:, 1:2]
    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)
    axis_max_dist = min(max_dist, fwd_min + fwd_span + margin + step)

    d_long_x = torch.cat([cos_h, -cos_h], dim=1)
    d_long_y = torch.cat([sin_h, -sin_h], dim=1)
    x_cross = axis_march_crossing(
        sample, bx, by, d_long_x, d_long_y, margin, step, axis_max_dist,
        interp_crossing)
    s_fwd = ((x_cross[:, 0:1] - fwd_min) / fwd_span).clamp(0.0, 1.0)
    s_bwd = ((x_cross[:, 1:2] - fwd_min) / fwd_span).clamp(0.0, 1.0)

    fx_local = fwd_min + s_fwd * fwd_span
    bwd_local = fwd_min + s_bwd * fwd_span
    sx_f = bx + fx_local * cos_h
    sy_f = by + fx_local * sin_h
    sx_b = bx - bwd_local * cos_h
    sy_b = by - bwd_local * sin_h
    start_x = torch.cat([sx_f, sx_f, sx_b, sx_b], dim=1)
    start_y = torch.cat([sy_f, sy_f, sy_b, sy_b], dim=1)
    d_lat_x = torch.cat([-sin_h, sin_h, -sin_h, sin_h], dim=1)
    d_lat_y = torch.cat([cos_h, -cos_h, cos_h, -cos_h], dim=1)
    y_cross = axis_march_crossing(
        sample, start_x, start_y, d_lat_x, d_lat_y, margin, step,
        axis_max_dist, interp_crossing)
    y_front = torch.minimum(y_cross[:, 0:1], y_cross[:, 1:2])
    s_fw = ((y_front - lat_min) / lat_span).clamp(0.0, 1.0)
    y_back = torch.minimum(y_cross[:, 2:3], y_cross[:, 3:4])
    s_bw = ((y_back - lat_min) / lat_span).clamp(0.0, 1.0)

    y_up_m = axis_march_crossing(
        sample, bx, by, -sin_h, cos_h, margin, step, axis_max_dist, interp_crossing)
    y_dn_m = axis_march_crossing(
        sample, bx, by, sin_h, -cos_h, margin, step, axis_max_dist, interp_crossing)
    y_mid = torch.minimum(y_up_m, y_dn_m)
    lat_min_mw = float(low[1])
    mw_span = float(high[1] - low[1])
    s_mw = ((y_mid - lat_min_mw) / mw_span).clamp(0.0, 1.0)

    return torch.cat([s_fw.clamp(0.0, 1.0), s_mw.clamp(0.0, 1.0),
                      s_bw.clamp(0.0, 1.0), s_fwd, s_bwd], dim=1)
