"""Bitwise-equivalence tests for the envelope-oracle ray marches.

These tests guard the oracle march optimisations (dead-code removal and
batching of the ``axis`` marches).  Every assertion is ``torch.equal`` -- no
tolerances -- because the optimisations are required to be *bitwise* identical
so that training remains reproducible.

The reference implementations live in ``_oracle_reference.py`` and are frozen
copies of the pre-optimisation code.  See that module for the rules about
updating them.

Why these tests exist
---------------------
All three bugs found while developing the optimisation were SILENT: no
exception, no crash, just quietly wrong scales (worst case max|diff| ~0.95 on
a [0, 1] scale).  Only bitwise comparison against a frozen reference catches
them.  ``tests/ea2/test_envelope_oracle.py`` uses ``pytest.approx`` with
1e-3..1e-4 tolerances and is therefore NOT sufficient for this purpose.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage

from legged_gym.envs.el_4090.envelope_adaptive_2 import envelope_oracle as eo
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    hex_body_sample_points,
)

try:  # pytest: package ``ea2``
    from . import _oracle_reference as ref
except ImportError:  # direct script execution
    import _oracle_reference as ref

_LOW = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9], dtype=torch.float32)
_HIGH = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6], dtype=torch.float32)
_RES = 0.1
_SIZE = 740
_WMIN = -37.0


# ---------------------------------------------------------------------------
# fixtures / scenario builders
# ---------------------------------------------------------------------------

def _field(seed: int = 0, n_tiles: int = 2, per_tile: int = 6) -> np.ndarray:
    """Pillar-field distance transform, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    boxes = []
    for tx in range(n_tiles):
        for ty in range(n_tiles):
            ox, oy = -32.0 + tx * 16.0, -32.0 + ty * 16.0
            for _ in range(per_tile):
                boxes.append((ox + 1.0 + rng.random() * 14.0,
                              oy + 1.0 + rng.random() * 14.0,
                              0.25 + rng.random() * 1.5,
                              0.25 + rng.random() * 1.5))
    xs = np.arange(_SIZE) * _RES + _WMIN
    XX, YY = np.meshgrid(xs, xs)
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    for cx, cy, hx, hy in boxes:
        mask |= (np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)
    return ndimage.distance_transform_edt(~mask, sampling=(_RES, _RES)).astype(np.float32)


@pytest.fixture(scope="module")
def field():
    return _field()


def _poses(n: int, seed: int, span: float = 20.0):
    g = torch.Generator().manual_seed(seed)
    head = torch.rand(n, generator=g) * 6.283185307179586 - 3.141592653589793
    pos = (torch.rand(n, 2, generator=g) * span - span / 2).contiguous()
    return head, pos


def _poses_inside_margin(field, n: int, margin: float = 0.10, seed: int = 0):
    """Start points whose clearance is ALREADY below ``margin``.

    This is the only family of poses that exercises the explicit ``t = 0``
    sample in ``_axis_march_crossing``.  Random path poses almost never hit it
    (0/4096 in one measurement), which is why a "t=0 dropped" regression can
    slip through a purely random test suite.
    """
    rng = np.random.default_rng(seed)
    cand = np.argwhere((field > 0.0) & (field < margin))
    if len(cand) == 0:                                   # pragma: no cover
        cand = np.argwhere(field < margin * 3)
    sel = cand[rng.choice(len(cand), size=min(n, len(cand)), replace=False)]
    pos = torch.tensor(np.stack([_WMIN + (sel[:, 1] + 0.5) * _RES,
                                 _WMIN + (sel[:, 0] + 0.5) * _RES], axis=-1),
                       dtype=torch.float32)
    head = torch.tensor(rng.uniform(-np.pi, np.pi, size=pos.shape[0]), dtype=torch.float32)
    return head, pos


# ---------------------------------------------------------------------------
# 1. the frozen reference really is the production behaviour
# ---------------------------------------------------------------------------

def test_reference_copy_matches_production(field):
    """If this fails, _oracle_reference.py is wrong and every other test here
    is meaningless.  Run it first."""
    head, pos = _poses(256, seed=7)
    for interp in (True, False):
        prod = eo._compute_raw_scales(head, pos, field, _LOW, _HIGH,
                                      0.10, 0.05, 5.0, interp, "axis")
        frozen = ref.raw_scales_axis(head, pos, field, _LOW, _HIGH,
                                     0.10, 0.05, 5.0, interp)
        assert torch.equal(prod, frozen), (
            f"frozen reference diverged from production (interp={interp})")


# ---------------------------------------------------------------------------
# 2. axis-mode equivalence (the production path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interp", [True, False])
def test_axis_matches_reference_random_poses(field, interp):
    head, pos = _poses(256, seed=11)
    prod = eo._compute_raw_scales(head, pos, field, _LOW, _HIGH,
                                  0.10, 0.05, 5.0, interp, "axis")
    frozen = ref.raw_scales_axis(head, pos, field, _LOW, _HIGH,
                                 0.10, 0.05, 5.0, interp)
    assert torch.equal(prod, frozen)


@pytest.mark.parametrize("interp", [True, False])
def test_axis_matches_reference_start_inside_margin(field, interp):
    """REGRESSION: guards the explicit t=0 sample in the axis march.

    A mutant that drops the t=0 sample passes every random-pose test but is
    caught here (interp=True).  Under interp=False the difference is absorbed
    by the ``clamp(0, 1)`` downstream, so this test cannot catch it in that
    mode -- documented limitation, harmless because production uses
    ``oracle_interp_crossing = True``.
    """
    head, pos = _poses_inside_margin(field, 192)
    prod = eo._compute_raw_scales(head, pos, field, _LOW, _HIGH,
                                  0.10, 0.05, 5.0, interp, "axis")
    frozen = ref.raw_scales_axis(head, pos, field, _LOW, _HIGH,
                                 0.10, 0.05, 5.0, interp)
    assert torch.equal(prod, frozen)


@pytest.mark.parametrize("interp", [True, False])
@pytest.mark.parametrize("margin,step,max_dist", [
    (0.10, 0.05, 5.0),     # production config -> 21 iterations
    (0.10, 0.01, 5.0),     # 105 iterations
    (0.10, 0.10, 5.0),     # 10 iterations
    (0.10, 0.30, 5.0),     # 3 iterations
    (0.10, 1.00, 5.0),     # 1 iteration
    (0.00, 0.05, 5.0),     # margin 0
    (0.50, 0.05, 5.0),     # margin larger than the march range
    (0.10, 0.05, 0.30),    # tiny max_dist
])
def test_axis_matches_reference_march_geometry(field, interp, margin, step, max_dist):
    """Sweeps the loop-trip count, including the degenerate small-T cases."""
    head, pos = _poses(128, seed=23)
    prod = eo._compute_raw_scales(head, pos, field, _LOW, _HIGH,
                                  margin, step, max_dist, interp, "axis")
    frozen = ref.raw_scales_axis(head, pos, field, _LOW, _HIGH,
                                 margin, step, max_dist, interp)
    assert torch.equal(prod, frozen)


def test_axis_handles_degenerate_and_hostile_inputs(field):
    """Empty batch, single env, float64 bounds, non-contiguous base_pos.

    ``base_pos[:, :2]`` is a strided view -- that is literally what the env
    passes, so it must keep working.
    """
    head, pos = _poses(64, seed=31)

    # empty batch
    out = eo._compute_raw_scales(head[:0], pos[:0], field, _LOW, _HIGH,
                                 0.10, 0.05, 5.0, True, "axis")
    assert out.shape == (0, 5)

    # single env
    p1 = eo._compute_raw_scales(head[:1], pos[:1], field, _LOW, _HIGH,
                                0.10, 0.05, 5.0, True, "axis")
    f1 = ref.raw_scales_axis(head[:1], pos[:1], field, _LOW, _HIGH,
                             0.10, 0.05, 5.0, True)
    assert torch.equal(p1, f1)

    # float64 bounds
    p64 = eo._compute_raw_scales(head, pos, field, _LOW.double(), _HIGH.double(),
                                 0.10, 0.05, 5.0, True, "axis")
    f64 = ref.raw_scales_axis(head, pos, field, _LOW.double(), _HIGH.double(),
                              0.10, 0.05, 5.0, True)
    assert torch.equal(p64, f64)

    # non-contiguous base_pos (the strided view the env actually passes)
    full = torch.cat([pos, torch.full((pos.shape[0], 1), 0.52)], dim=-1)
    assert not full[:, :2].is_contiguous()
    ps = eo._compute_raw_scales(head, full[:, :2], field, _LOW, _HIGH,
                                0.10, 0.05, 5.0, True, "axis")
    fs = ref.raw_scales_axis(head, full[:, :2], field, _LOW, _HIGH,
                             0.10, 0.05, 5.0, True)
    assert torch.equal(ps, fs)


def test_axis_equivalent_over_a_multi_step_sequence(field):
    """Robot moving along a trajectory: no state may leak between calls."""
    n = 32
    for i in range(25):
        pos = torch.stack([torch.full((n,), -5.0 + i * 0.25),
                           torch.linspace(-2.0, 2.0, n)], dim=-1)
        head = torch.full((n,), 0.3 * i)
        prod = eo._compute_raw_scales(head, pos, field, _LOW, _HIGH,
                                      0.10, 0.05, 5.0, True, "axis")
        frozen = ref.raw_scales_axis(head, pos, field, _LOW, _HIGH,
                                     0.10, 0.05, 5.0, True)
        assert torch.equal(prod, frozen), f"diverged at step {i}"


# ---------------------------------------------------------------------------
# 3. coupled mode must keep working (tests import it directly)
# ---------------------------------------------------------------------------

def test_coupled_mode_still_matches_its_own_semantics(field):
    """The axis optimisation must not touch the coupled branch.

    ``test_envelope_oracle.py`` compares coupled against axis, so both paths
    must remain correct even though production runs axis.
    """
    head, pos = _poses(64, seed=41)
    for interp in (True, False):
        out = eo._compute_raw_scales(head, pos, field, _LOW, _HIGH,
                                     0.10, 0.05, 5.0, interp, "coupled")
        assert out.shape == (64, 5)
        assert torch.isfinite(out).all()
        assert bool(((out >= 0.0) & (out <= 1.0)).all())


@pytest.mark.xfail(
    reason="group_mode validation is added with the axis dead-code removal; "
           "until then an unknown mode silently falls through to coupled",
    strict=True,
)
def test_unknown_group_mode_is_rejected(field):
    """An unknown group_mode must raise rather than silently return a coupled
    result.  Remove the xfail mark once the axis branch validates its input."""
    head, pos = _poses(8, seed=43)
    with pytest.raises(ValueError):
        eo._compute_raw_scales(head, pos, field, _LOW, _HIGH,
                               0.10, 0.05, 5.0, True, "bogus")


# ---------------------------------------------------------------------------
# 4. sanity: the axis branch is reachable and not a no-op
# ---------------------------------------------------------------------------

def test_axis_branch_produces_different_scales_than_coupled(field):
    """Guards against the branch silently falling through to coupled -- the
    single most dangerous failure mode (max|diff| ~0.95, silent)."""
    head, pos = _poses(128, seed=53)
    cpl = eo._compute_raw_scales(head, pos, field, _LOW, _HIGH,
                                 0.10, 0.05, 5.0, True, "coupled")
    axi = eo._compute_raw_scales(head, pos, field, _LOW, _HIGH,
                                 0.10, 0.05, 5.0, True, "axis")
    # they must be genuinely different implementations, not accidentally equal
    assert not torch.equal(cpl, axi)
    # ...but both finite and in range
    for t in (cpl, axi):
        assert torch.isfinite(t).all()
        assert bool(((t >= 0.0) & (t <= 1.0)).all())
