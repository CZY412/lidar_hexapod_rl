"""Gate G5 -- can PPO pick up where supervised learning left off?

This is the acceptance test for the whole pipeline: the SL weights are only
useful if a PPO run can continue training from them.

Thresholds are asserted from the JSON files produced by
``sl/scripts/ppo_continue.py`` when ``EA2_SL_PPO_DIR`` points at a directory
containing ``ppo_scratch.json`` and at least one ``ppo_sl_init*.json``.
"""

from __future__ import annotations

import json
import math
import os

import pytest

PPO_DIR = os.environ.get("EA2_SL_PPO_DIR", "")


def _load(name: str) -> dict:
    path = os.path.join(PPO_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    with open(path) as f:
        return json.load(f)


def _at(curve, it: int):
    for pt in curve:
        if pt["iter"] == it:
            return pt["step_reward"]
    return None


@pytest.mark.skipif(not PPO_DIR, reason="EA2_SL_PPO_DIR not set")
def test_all_curves_are_finite():
    for name in ("ppo_scratch.json", "ppo_slinits1.json", "ppo_slinits42.json"):
        data = _load(name)
        for pt in data["curve"]:
            assert math.isfinite(pt["step_reward"]), f"{name} has non-finite reward at iter {pt['iter']}"


@pytest.mark.skipif(not PPO_DIR, reason="EA2_SL_PPO_DIR not set")
def test_sl_init_starts_ahead_of_scratch():
    """The whole point: a better starting point."""
    scratch = _load("ppo_scratch.json")
    sl = _load("ppo_slinits1.json")
    s0, p0 = scratch["curve"][0], sl["curve"][0]
    print(f"[G5] iter1  scratch={s0['step_reward']:+.4f}  sl_init={p0['step_reward']:+.4f}")
    assert p0["step_reward"] > s0["step_reward"], "SL init does not improve the starting point"
    gain = (p0["step_reward"] - s0["step_reward"]) / abs(s0["step_reward"])
    print(f"[G5] starting-point gain = {gain * 100:.1f}%")
    assert gain >= 0.15, f"starting-point gain too small: {gain * 100:.1f}%"


@pytest.mark.skipif(not PPO_DIR, reason="EA2_SL_PPO_DIR not set")
def test_sl_init_still_ahead_at_the_end():
    scratch = _load("ppo_scratch.json")
    sl = _load("ppo_slinits1.json")
    s_end, p_end = scratch["curve"][-1], sl["curve"][-1]
    print(f"[G5] final  scratch={s_end['step_reward']:+.4f}  sl_init={p_end['step_reward']:+.4f}")
    assert p_end["step_reward"] >= s_end["step_reward"] - 0.02, (
        "SL-initialised run fell behind scratch by more than tolerance"
    )


@pytest.mark.skipif(not PPO_DIR, reason="EA2_SL_PPO_DIR not set")
def test_advantage_is_weights_not_path_memorisation():
    """The cross-seed arm isolates weight quality from path familiarity.

    ``task_registry`` resets every RNG from ``cfg.seed``, so training PPO on a
    seed the SL model never saw removes any path-memorisation benefit.  If the
    advantage survives, it comes from the weights.
    """
    sl_same = _load("ppo_slinits1.json")
    sl_cross = _load("ppo_slinits42.json")
    a, b = sl_same["curve"][0]["step_reward"], sl_cross["curve"][0]["step_reward"]
    print(f"[G5] iter1  sl_init(seen map)={a:+.4f}  sl_init(unseen map)={b:+.4f}")
    assert b >= a - 0.05, (
        f"cross-seed start much worse ({a:+.4f} -> {b:+.4f}); the advantage may "
        "be path memorisation rather than transferable weights"
    )


@pytest.mark.skipif(not PPO_DIR, reason="EA2_SL_PPO_DIR not set")
def test_no_early_regression_spike():
    """SL init must not first destroy the pretrained behaviour and then recover."""
    sl = _load("ppo_slinits1.json")
    curve = sl["curve"]
    start = curve[0]["step_reward"]
    worst = min(pt["step_reward"] for pt in curve)
    print(f"[G5] sl_init start={start:+.4f}  worst={worst:+.4f}")
    assert worst >= start - 0.12, f"large regression spike: {start:+.4f} -> {worst:+.4f}"
