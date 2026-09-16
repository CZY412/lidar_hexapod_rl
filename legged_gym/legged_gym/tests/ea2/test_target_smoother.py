"""Unit tests for the production target smoother (target_smoother)."""

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import torch
import pytest

from legged_gym.envs.el_4090.envelope_adaptive_2.target_smoother import (
    RateLimitedOracle,
)

LOW = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9])
HIGH = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6])
N = 4
DT = 0.02


MIN_V = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.6])  # all extents = 0


def _smoother(**kw):
    defaults = dict(
        num_envs=N, dt=DT, device="cpu", low=LOW, high=HIGH,
        shrink_rate=2.0, grow_rate=0.5, cooldown_seconds=0.2,
    )
    defaults.update(kw)
    return RateLimitedOracle(**defaults)


def test_step_size_matches_physical_rate_spec():
    rl = _smoother()
    # shrink_rate 2.0 m/s at dt=0.02 -> 0.04 m per call -> extent step 0.04/span
    assert torch.allclose(rl.shrink_n, 0.04 / rl.span.abs(), atol=1e-6)
    # grow_rate 0.5 m/s -> 0.01 m per call; cooldown 0.2 s -> 10 calls
    assert torch.allclose(rl.grow_n, 0.01 / rl.span.abs(), atol=1e-6)
    assert rl.cooldown_calls == 10


def test_shrink_rate_limited_toward_raw():
    rl = _smoother()
    target = MIN_V.unsqueeze(0).expand(N, -1).clone()  # demand full shrink
    first = rl.update(target)
    # prev starts fully open (extent 1); one call shrinks by exactly shrink_n
    extent = ((first - rl.min_v) / rl.span)
    assert torch.allclose(extent, 1.0 - rl.shrink_n, atol=1e-5)


def test_shrink_converges_and_never_overshoots():
    rl = _smoother()
    target = MIN_V.unsqueeze(0).expand(N, -1).clone()
    prev_extent = torch.ones(N, 5)
    for _ in range(40):
        out = rl.update(target)
        extent = ((out - rl.min_v) / rl.span).clamp(0, 1)
        # monotone decrease, never below the raw target
        assert bool((extent <= prev_extent + 1e-6).all())
        prev_extent = extent
    assert torch.allclose(out, target, atol=1e-5)


def test_grow_requires_cooldown_then_rate_limited():
    rl = _smoother()
    # first collapse all extents to 0
    for _ in range(80):
        rl.update(MIN_V.unsqueeze(0).expand(N, -1).clone())
    # now demand full open: held until cooldown_calls consecutive clears
    target = torch.stack([HIGH[0], HIGH[1], HIGH[2], HIGH[3], LOW[4]]).unsqueeze(0).expand(N, -1).clone()
    out = rl.update(target)
    for _ in range(rl.cooldown_calls - 2):
        out = rl.update(target)  # held: counter < cooldown_calls
    extent = (out - rl.min_v) / rl.span
    assert float(extent.abs().max()) < 1e-5  # still held during cooldown
    out_after = rl.update(target)  # counter reaches cooldown -> grows grow_n
    extent_after = (out_after - rl.min_v) / rl.span
    assert torch.allclose((extent_after - extent).abs(), rl.grow_n, atol=1e-6)


def test_per_env_safety_snap_isolates_violating_env():
    def check(cand):
        # env 0 declared unsafe
        return torch.tensor([True, False, False, False])

    rl = _smoother(safety_check=check)
    raw = torch.stack([
        LOW,                       # env 0 raw (safe fallback)
        torch.tensor([0.45, 0.5, 0.45, 0.75, -0.75]),
        torch.tensor([0.45, 0.5, 0.45, 0.75, -0.75]),
        torch.tensor([0.45, 0.5, 0.45, 0.75, -0.75]),
    ])
    out = rl.update(raw)
    assert bool(rl.snapped[0]) and not bool(rl.snapped[1])
    # env 0 snapped to raw; envs 1..3 got the rate-limited candidate
    assert torch.allclose(out[0], raw[0])
    assert not torch.allclose(out[1], raw[1])
    # snapped env state re-seeded from raw: next call continues from there
    assert torch.allclose(rl.prev_s[0], rl._to_s(raw[0]).clamp(0, 1), atol=1e-5)


def test_reset_ids_reinitialises_selected_envs():
    rl = _smoother()
    for _ in range(40):
        rl.update(MIN_V.unsqueeze(0).expand(N, -1).clone())  # all collapsed
    assert float(rl.prev_s[1].max()) < 0.5
    rl.reset_ids([1])
    assert torch.allclose(rl.prev_s[1], torch.ones(5))
    assert torch.allclose(rl.counter[1], torch.zeros(5))
    # other envs keep their collapsed state
    assert float(rl.prev_s[0].max()) < 0.5


def test_signed_span_direction_uniformity():
    """Extent space must treat backward_limit physically: s=0 shrunk rear."""
    rl = _smoother()
    full_open = torch.stack([HIGH[0], HIGH[1], HIGH[2], HIGH[3], LOW[4]]).unsqueeze(0).expand(N, -1).clone()
    out = rl.update(full_open)  # already at max extent -> unchanged (held at open)
    # backward_limit at max extent is the most negative value (-0.9)
    assert float(out[0, 4]) == pytest.approx(-0.9, abs=1e-5)
