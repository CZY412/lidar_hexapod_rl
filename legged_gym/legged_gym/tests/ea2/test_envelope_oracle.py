"""Tests for ``envelope_oracle.compute_oracle_params``.

These tests exercise the grid-distance-based theoretical envelope generator on
synthetic distance fields.  They do not create an Isaac Gym simulation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage

from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
    GROUP_INDICES,
    PARAM_NAMES,
    compute_direct_oracle_params_with_stats,
    compute_oracle_params,
)

_LOW = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9], dtype=torch.float32)
_HIGH = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6], dtype=torch.float32)
_MAX_V = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.9], dtype=torch.float32)
_MIN_V = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.6], dtype=torch.float32)


def _open_field():
    return np.full((740, 740), 10.0, dtype=np.float32)


def test_group_indices_cover_all_boundary_points():
    """Every boundary point should belong to at least one parameter group."""
    all_idx = set()
    for idxs in GROUP_INDICES.values():
        all_idx.update(idxs)
    assert all_idx == set(range(24)), sorted(all_idx - set(range(24)))


def test_open_field_returns_physical_max():
    """With no obstacles nearby the oracle should be the physical maximum."""
    oracle = compute_oracle_params(
        torch.zeros(1),
        torch.zeros(1, 2),
        _open_field(),
        _LOW,
        _HIGH,
    )
    assert oracle.shape == (1, 5)
    torch.testing.assert_close(oracle[0], _MAX_V, atol=1e-4, rtol=1e-4)


def test_front_obstacle_reduces_forward_limit():
    """An obstacle directly ahead should shrink forward_limit, not widths."""
    df = _open_field()
    # Grid cell near world x=0.80, y=0 (ix=377, iy=370).
    df[370, 377] = 0.0
    oracle = compute_oracle_params(
        torch.zeros(1),
        torch.zeros(1, 2),
        df,
        _LOW,
        _HIGH,
    )
    assert oracle[0, 3].item() < _MAX_V[3].item()
    assert oracle[0, 0].item() == pytest.approx(_MAX_V[0].item(), abs=1e-4)
    assert oracle[0, 1].item() == pytest.approx(_MAX_V[1].item(), abs=1e-4)
    assert oracle[0, 4].item() == pytest.approx(_MAX_V[4].item(), abs=1e-4)


def test_side_obstacle_reduces_width_somewhat():
    """A side obstacle should reduce at least one width parameter."""
    df = _open_field()
    # Put an obstacle to the left at approx x=0, y=0.8 (ix=370, iy=378?) 
    # Grid rows are y, cols x; y=0.8 -> iy=377, x=0 -> ix=370.
    df[377, 370] = 0.0
    oracle = compute_oracle_params(
        torch.zeros(1),
        torch.zeros(1, 2),
        df,
        _LOW,
        _HIGH,
    )
    assert oracle[0, :3].max().item() < _MAX_V[:3].max().item() + 1e-6


def test_oracle_stays_in_bounds():
    """Oracle params must stay inside the legal physical envelope bounds."""
    df = _open_field()
    df[370, 377] = 0.0
    oracle = compute_oracle_params(
        torch.zeros(4),
        torch.zeros(4, 2),
        df,
        _LOW,
        _HIGH,
    )
    assert oracle.shape == (4, 5)
    assert torch.isfinite(oracle).all()
    # Check physical bounds: widths/fwd in [low, high], backward in [high, low].
    assert bool((oracle[:, :4] >= _LOW[:4].unsqueeze(0) - 1e-6).all())
    assert bool((oracle[:, :4] <= _HIGH[:4].unsqueeze(0) + 1e-6).all())
    assert bool((oracle[:, 4] >= _LOW[4].unsqueeze(0) - 1e-6).all())
    assert bool((oracle[:, 4] <= _HIGH[4].unsqueeze(0) + 1e-6).all())


@pytest.mark.parametrize("group_mode", ["coupled", "axis"])
def test_direct_oracle_open_field_soft_max(group_mode):
    """Direct oracle on open field should stay at the soft maximum.

    Both group modes must agree here: with no obstacle there is nothing to
    decouple, so the envelope sits at the soft maximum in every dimension.
    """
    direct, _ = compute_direct_oracle_params_with_stats(
        torch.zeros(1),
        torch.zeros(1, 2),
        _open_field(),
        _LOW,
        _HIGH,
        group_mode=group_mode,
    )
    margin = (1.0 - 0.9) * (_HIGH - _LOW) / 2.0
    soft_high = _HIGH - margin
    soft_low = _LOW + margin
    # forward params at soft upper bound; backward at soft lower bound (largest
    # rear extent within soft range).
    assert bool((direct[0, :4] >= soft_high[:4] - 1e-4).all())
    assert direct[0, 4].item() == pytest.approx(soft_low[4].item(), abs=1e-4)


@pytest.mark.parametrize("group_mode", ["coupled", "axis"])
def test_direct_oracle_front_obstacle_only_shrinks_forward(group_mode):
    """With a front obstacle the envelope must shrink forward and stay safe.

    ``coupled`` additionally guarantees that widths are not dragged down by a
    shared radial multiplier -- that was the historical "global multiplier"
    bug this test guards.  ``axis`` decouples the parameters, so with
    ``interp_crossing=False`` it may legitimately shrink ``front_width``
    further (its vertical march sees the obstacle at the forward apex).  What
    both modes must guarantee is that the result is collision free.

    Note the front obstacle sits *inside* the unconstrained hexagon, so the
    boundary-only collision check is what the oracle is defined against.
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
        hex_collision_terms,
    )

    df = _open_field()
    df[370, 377] = 0.0
    head = torch.zeros(1)
    pos = torch.zeros(1, 2)
    direct, _ = compute_direct_oracle_params_with_stats(
        head,
        pos,
        df,
        _LOW,
        _HIGH,
        group_mode=group_mode,
    )
    margin = (1.0 - 0.9) * (_HIGH - _LOW) / 2.0
    soft_high = _HIGH - margin
    assert direct[0, 3].item() < soft_high[3].item() - 1e-4

    # Safety is mode independent: the returned envelope must be collision free.
    _, hard = hex_collision_terms(
        direct, head, pos, torch.as_tensor(df), margin=0.10, soft_margin=0.10
    )
    assert float(hard.max()) <= 0.06

    if group_mode == "coupled":
        # widths stay near soft max (no global multiplier dragging them down)
        assert direct[0, 0].item() == pytest.approx(soft_high[0].item(), abs=1e-4)
        assert direct[0, 1].item() == pytest.approx(soft_high[1].item(), abs=1e-4)
        assert direct[0, 2].item() == pytest.approx(soft_high[2].item(), abs=1e-4)
    else:
        # axis: middle/back widths are unconstrained by a front obstacle, only
        # front_width may follow the shrinking forward apex.
        assert direct[0, 1].item() == pytest.approx(soft_high[1].item(), abs=1e-4)
        assert direct[0, 2].item() == pytest.approx(soft_high[2].item(), abs=1e-4)
        assert direct[0, 0].item() <= soft_high[0].item() + 1e-4


def test_param_names_order():
    assert PARAM_NAMES == (
        "front_width",
        "middle_width",
        "back_width",
        "forward_limit",
        "backward_limit",
    )


# ---------------------------------------------------------------------------
# interp_crossing=True (the production reward path since
# ``envelope.oracle_interp_crossing = True``) -- regression guards for the
# bilinear-sampling + interpolated-crossing branch
# ---------------------------------------------------------------------------


def _corridor_field(half_width: float) -> np.ndarray:
    mask = np.zeros((740, 740), dtype=bool)
    ys = np.arange(740) * 0.1 - 37.0
    for iy, y in enumerate(ys):
        if abs(y) > half_width:
            mask[iy, :] = True
    return ndimage.distance_transform_edt(
        ~mask, sampling=(0.1, 0.1)
    ).astype(np.float32)


def _extent(params: torch.Tensor) -> torch.Tensor:
    """Signed-span normalised extents (0 = fully shrunk, 1 = fully open)."""
    span = _MAX_V - _MIN_V  # backward component is negative
    return ((params - _MIN_V) / span).clamp(0.0, 1.0)


def _area(params: torch.Tensor) -> float:
    """Hexagon area (m^2): two congruent triangles sharing the lateral axis.

    front triangle  = front_width * forward_limit
    rear triangle   = back_width * |backward_limit|
    ``backward_limit`` is negative, hence the minus sign.
    """
    return float(params[0] * params[3] - params[4] * params[2])


@pytest.mark.parametrize("group_mode", ["coupled", "axis"])
def test_interp_open_field_matches_nearest_and_soft_max(group_mode):
    """In open space both crossing modes must agree and stay at soft max."""
    head = torch.zeros(1)
    pos = torch.zeros(1, 2)
    df = _open_field()
    near, _ = compute_direct_oracle_params_with_stats(
        head, pos, df, _LOW, _HIGH, interp_crossing=False, group_mode=group_mode
    )
    itp, _ = compute_direct_oracle_params_with_stats(
        head, pos, df, _LOW, _HIGH, interp_crossing=True, group_mode=group_mode
    )
    torch.testing.assert_close(near, itp)
    margin = (1.0 - 0.9) * (_HIGH - _LOW) / 2.0
    torch.testing.assert_close(itp[0, :4], (_HIGH - margin)[:4], atol=1e-4, rtol=0)
    assert itp[0, 4].item() == pytest.approx((_LOW + margin)[4].item(), abs=1e-4)


@pytest.mark.parametrize("group_mode", ["coupled", "axis"])
def test_interp_never_tightens_and_stays_safe_on_corridor(group_mode):
    """Removing the one-step back-off may only relax the envelope, and both
    modes must satisfy the active-set safety post-condition."""
    head = torch.zeros(1)
    pos = torch.zeros(1, 2)
    df = _corridor_field(0.65)
    near, _ = compute_direct_oracle_params_with_stats(
        head, pos, df, _LOW, _HIGH, interp_crossing=False, group_mode=group_mode
    )
    itp, _ = compute_direct_oracle_params_with_stats(
        head, pos, df, _LOW, _HIGH, interp_crossing=True, group_mode=group_mode
    )
    # Relaxation only: the interpolated crossing must not yield a smaller
    # envelope overall.  Total hexagon area is the mode-independent measure --
    # ``coupled`` additionally satisfies per-dimension monotonicity, but
    # ``axis`` solves the parameters sequentially, so relaxing
    # ``forward_limit`` moves the forward apex outward and can therefore
    # tighten ``middle_width`` at the new apex.  The net area still grows.
    assert _area(itp[0]) >= _area(near[0]) - 1e-5
    if group_mode == "coupled":
        # per-dimension: interp extents >= nearest extents (backward_limit's
        # signed span makes the comparison direction-agnostic)
        assert bool((_extent(itp) >= _extent(near) - 1e-5).all())
    # safety post-condition: active-set keeps worst boundary violation <=
    # its 0.05 threshold (plus eps)
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
        hex_collision_terms,
    )

    dft = torch.as_tensor(df)
    for params in (near, itp):
        _, hard = hex_collision_terms(
            params, head, pos, dft, margin=0.10, soft_margin=0.10
        )
        assert float(hard.max()) <= 0.06


def test_interp_crossing_exact_on_linear_ramp():
    """On a field whose distance ramps exactly linearly in x, the bilinear
    interpolated crossing is exact: front_width/forward_limit scale must be
    (wall_x - robot_x) / full_hex_x_extent, hand-computable.

    value(x) = 0.5 * (37 - x)  ->  margin contour at x = 36.8;
    robot at x = 36.0, full hex x extent 0.9  ->  scale = 0.8 / 0.9.
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
        _compute_raw_scales,
    )

    xs = -37.0 + (np.arange(740) + 0.5) * 0.1
    values = (0.5 * (37.0 - xs)).astype(np.float32)
    df = np.broadcast_to(values[None, :], (740, 740)).copy()

    head = torch.zeros(1)
    pos = torch.tensor([[36.0, 0.0]])
    s_itp = _compute_raw_scales(
        head, pos, df, _LOW, _HIGH, margin=0.10, step=0.05, max_dist=5.0,
        interp_crossing=True,
    )
    expected = 0.8 / 0.9
    assert s_itp[0, 0].item() == pytest.approx(expected, abs=1e-3)
    assert s_itp[0, 3].item() == pytest.approx(expected, abs=1e-3)
    # geometry is open backward/sideways on this ramp
    assert float(s_itp[0, 1]) == pytest.approx(1.0, abs=1e-4)
    assert float(s_itp[0, 2]) == pytest.approx(1.0, abs=1e-4)
    assert float(s_itp[0, 4]) == pytest.approx(1.0, abs=1e-4)

    # nearest-cell mode stays quantised inside the analytic back-off window
    s_near = _compute_raw_scales(
        head, pos, df, _LOW, _HIGH, margin=0.10, step=0.05, max_dist=5.0,
        interp_crossing=False,
    )
    assert 0.84 <= s_near[0, 0].item() <= 0.9539


@pytest.mark.parametrize("group_mode", ["coupled", "axis"])
def test_soft_floor_yields_to_safety_in_infeasible_corridor(group_mode):
    """REGRESSION (soft-cap ordering): when the active-set shrink must go
    below a per-parameter soft floor for safety, safety wins.

    In a 0.25 m half-width corridor even the minimum envelope violates, so
    every parameter must collapse to its HARD bound (middle_width -> 0.3,
    below its 0.32 soft floor).  The old ordering (full soft clamp applied
    AFTER the shrink) raised the widths back to their soft floors.

    Both group modes are exercised: ``axis`` decouples the parameters, so the
    lengths could in principle stay open, but an *infeasible* corridor must
    still collapse everything to the hard bound.
    """
    head = torch.zeros(1)
    pos = torch.zeros(1, 2)
    df = _corridor_field(0.25)
    direct, _ = compute_direct_oracle_params_with_stats(
        head, pos, df, _LOW, _HIGH, interp_crossing=True, group_mode=group_mode
    )
    torch.testing.assert_close(direct[0], _MIN_V, atol=1e-5, rtol=0)
    # the discriminating assertion: below the soft floor
    soft_margin = (1.0 - 0.9) * (_HIGH - _LOW) / 2.0
    assert bool((direct[0, :3] < (_LOW + soft_margin)[:3] - 1e-4).all())


@pytest.mark.parametrize("group_mode", ["coupled", "axis"])
def test_direct_output_within_hard_bounds_under_aggressive_shrink(group_mode):
    """Even when the active-set saturates, outputs stay within the hard
    physical bounds."""
    head = torch.zeros(2)
    pos = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
    df = _corridor_field(0.25)
    direct, raw_hard = compute_direct_oracle_params_with_stats(
        head, pos, df, _LOW, _HIGH, interp_crossing=True, group_mode=group_mode
    )
    assert bool((direct[:, :4] >= _LOW[:4].unsqueeze(0) - 1e-5).all())
    assert bool((direct[:, :4] <= _HIGH[:4].unsqueeze(0) + 1e-5).all())
    assert bool((direct[:, 4] >= _LOW[4].unsqueeze(0) - 1e-5).all())
    assert bool((direct[:, 4] <= _HIGH[4].unsqueeze(0) + 1e-5).all())
    assert torch.isfinite(direct).all() and torch.isfinite(raw_hard).all()


def test_input_validation_errors():
    head = torch.zeros(1)
    pos = torch.zeros(1, 2)
    df = _open_field()
    with pytest.raises(ValueError):
        compute_oracle_params(head, pos, None, _LOW, _HIGH)
    with pytest.raises(ValueError):
        compute_oracle_params(head, pos, df, _LOW, _HIGH, step=0.0)
    with pytest.raises(ValueError):
        compute_oracle_params(head, pos, df, _LOW, _HIGH, max_dist=-1.0)
    with pytest.raises(ValueError):
        compute_oracle_params(head, pos, df, _LOW[:4], _HIGH)


def test_physical_min_max_backward_reversal():
    """The rear extent is reversed: more negative backward_limit = larger."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
        _physical_min_max,
    )

    min_v, max_v = _physical_min_max(_LOW, _HIGH)
    torch.testing.assert_close(min_v[:4], _LOW[:4])
    torch.testing.assert_close(max_v[:4], _HIGH[:4])
    # backward: physically smallest extent = -0.6 (= _HIGH[4])
    assert min_v[4].item() == pytest.approx(_HIGH[4].item())
    assert max_v[4].item() == pytest.approx(_LOW[4].item())


def test_direct_boundary_groups_cover_all_samples():
    """Every boundary sample must influence at least one direct-oracle
    parameter, and the known front/back group coupling is pinned (see the
    group table: front_width shares forward_limit's samples by design)."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
        _DIRECT_BOUNDARY_GROUPS,
        _SAMPLE_TO_PARAM_MASK,
    )

    union: set = set()
    for idxs in _DIRECT_BOUNDARY_GROUPS.values():
        union.update(idxs)
    assert union == set(range(24))
    assert bool(_SAMPLE_TO_PARAM_MASK.any(dim=1).all())
    # pinned coupling (documented limitation: per-parameter independence is
    # not complete for these pairs)
    assert (
        _DIRECT_BOUNDARY_GROUPS["front_width"]
        == _DIRECT_BOUNDARY_GROUPS["forward_limit"]
    )
    assert (
        _DIRECT_BOUNDARY_GROUPS["back_width"]
        == _DIRECT_BOUNDARY_GROUPS["backward_limit"]
    )


def test_raw_scale_monotone_under_narrowing():
    """Narrowing the corridor must not increase any per-parameter scale."""
    head = torch.zeros(1)
    pos = torch.zeros(1, 2)
    prev = None
    for half_width in (1.0, 0.7, 0.5):
        from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
            _compute_raw_scales,
        )

        scales = _compute_raw_scales(
            head, pos, _corridor_field(half_width), _LOW, _HIGH,
            margin=0.10, step=0.05, max_dist=5.0, interp_crossing=True,
        )
        if prev is not None:
            assert bool((scales <= prev + 1e-5).all())
        prev = scales


# ---------------------------------------------------------------------------
# group_mode="axis" -- parameter-group decoupling (verified coordinate
# decomposition: the A-B edge is the line x = forward_limit, y = +-front_width
# ---------------------------------------------------------------------------


def _thick_wall_field_x(x_wall: float, cells: int = 5) -> np.ndarray:
    mask = np.zeros((740, 740), dtype=bool)
    ix_w = int(round((x_wall + 37.0) / 0.1))
    mask[:, ix_w : ix_w + cells] = True
    return ndimage.distance_transform_edt(~mask, sampling=(0.1, 0.1)).astype(np.float32)


def test_axis_mode_front_wall_opens_width():
    """Dead-ahead wall: axis mode shrinks forward_limit and leaves
    front_width fully open (the coupled mode drags it down)."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
        _compute_raw_scales,
    )

    head = torch.zeros(1)
    pos = torch.zeros(1, 2)
    df = _thick_wall_field_x(0.8)
    s_cpl = _compute_raw_scales(
        head, pos, df, _LOW, _HIGH, 0.10, 0.05, 5.0, True, "coupled"
    )
    s_ax = _compute_raw_scales(
        head, pos, df, _LOW, _HIGH, 0.10, 0.05, 5.0, True, "axis"
    )
    # both constrain the forward extent
    assert float(s_ax[0, 3]) < 1.0
    # decoupling: the width opens under axis mode
    assert float(s_ax[0, 0]) == pytest.approx(1.0, abs=1e-5)
    assert float(s_cpl[0, 0]) < float(s_ax[0, 0]) - 1e-3
    # rear is open in both
    assert float(s_ax[0, 4]) == pytest.approx(1.0, abs=1e-4)


def test_axis_mode_corridor_constrains_widths_not_forward():
    """Parallel walls: axis mode constrains the widths, forward stays open."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
        _compute_raw_scales,
    )

    mask = np.zeros((740, 740), dtype=bool)
    ys = np.arange(740) * 0.1 - 37.0
    for iy, y in enumerate(ys):
        if abs(y) > 0.65:
            mask[iy, :] = True
    df = ndimage.distance_transform_edt(~mask, sampling=(0.1, 0.1)).astype(np.float32)
    head = torch.zeros(1)
    pos = torch.zeros(1, 2)
    s_ax = _compute_raw_scales(
        head, pos, df, _LOW, _HIGH, 0.10, 0.05, 5.0, True, "axis"
    )
    # lateral axes hit the walls; mw is exact at the vertical march
    assert float(s_ax[0, 0]) < 1.0 and float(s_ax[0, 1]) < 1.0 and float(s_ax[0, 2]) < 1.0
    # margin contour: mw_max = wall(0.65) - margin(0.10) = 0.55
    assert s_ax[0, 1].item() == pytest.approx((0.65 - 0.10 - 0.3) / 0.4, abs=1e-3)
    # nothing ahead of the robot along +x within 5 m
    assert float(s_ax[0, 3]) == pytest.approx(1.0, abs=1e-6)
    assert float(s_ax[0, 4]) == pytest.approx(1.0, abs=1e-6)


def test_axis_mode_ramp_exact_values():
    """On the y-independent linear ramp the axis-mode forward scale is exactly
    (contour_x - robot_x) / forward_span and the width is unconstrained."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
        _compute_raw_scales,
    )

    xs = -37.0 + (np.arange(740) + 0.5) * 0.1
    values = (0.5 * (37.0 - xs)).astype(np.float32)
    df = np.broadcast_to(values[None, :], (740, 740)).copy()
    head = torch.zeros(1)
    pos = torch.tensor([[36.0, 0.0]])
    s_ax = _compute_raw_scales(
        head, pos, df, _LOW, _HIGH, 0.10, 0.05, 5.0, True, "axis"
    )
    assert s_ax[0, 3].item() == pytest.approx((0.8 - 0.6) / 0.3, abs=1e-3)
    assert float(s_ax[0, 0]) == pytest.approx(1.0, abs=1e-4)
    assert float(s_ax[0, 1]) == pytest.approx(1.0, abs=1e-4)


def test_axis_mode_direct_output_safety_postcondition():
    """Diagonal obstacle (axis marches are blind to it): the active-set stage
    must still return a safe composed hexagon within hard bounds."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
        hex_collision_terms,
    )

    df = _open_field()
    # diagonal pillar at (0.85, 0.85): visible to the front-width rays'
    # diagonal neighbours but not to the +-x axis marches
    df[377, 378] = 0.0
    head = torch.zeros(1)
    pos = torch.zeros(1, 2)
    direct, _ = compute_direct_oracle_params_with_stats(
        head, pos, df, _LOW, _HIGH, interp_crossing=True, group_mode="axis"
    )
    _, hard = hex_collision_terms(
        direct, head, pos, torch.as_tensor(df), margin=0.10, soft_margin=0.10
    )
    assert float(hard.max()) <= 0.06
    assert bool((direct[:, :4] >= _LOW[:4].unsqueeze(0) - 1e-5).all())
    assert bool((direct[:, 4] >= _LOW[4] - 1e-5) and (direct[:, 4] <= _HIGH[4] + 1e-5))
