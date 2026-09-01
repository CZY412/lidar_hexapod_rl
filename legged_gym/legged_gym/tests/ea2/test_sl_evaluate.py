"""Gate G3 -- closed-loop behaviour of the deployed supervised policy.

Thresholds are asserted from the JSON written by ``sl/scripts/eval.py`` when
``EA2_SL_EVAL`` points at it.  The pure-function tests below run without a GPU.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import s_to_action, load_checkpoint
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import (
    SLConfig,
    env_action_scale,
)

EVAL_JSON = os.environ.get("EA2_SL_EVAL", "")


# --------------------------------------------------------------------------
# action mapping
# --------------------------------------------------------------------------


def test_s_to_action_centre_is_zero():
    s = torch.full((4, 5), 0.5)
    a = s_to_action(s)
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-6)


def test_s_to_action_roundtrip_through_env_convention():
    """a -> target via the env formula must reproduce the s=0 / s=1 endpoints.

    Note the endpoint convention: ``s`` is normalised against
    ``MIN_V = [low[0..3], high[4]]`` and ``MAX_V = [high[0..3], low[4]]``,
    because ``backward_limit``'s admissible range runs from -0.9 (most rear
    extent, s=1) up to -0.6 (least, s=0).  So s=0 maps to ``MIN_V``, not to
    ``low``.  The action scale must come from the *live* env config
    (``env_action_scale``), not a hardcoded 0.9-based constant -- the fold and
    the env mapping drifted apart once ``soft_dof_pos_limit`` changed.
    """
    low = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9])
    high = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6])
    min_v = torch.stack([low[0], low[1], low[2], low[3], high[4]])
    max_v = torch.stack([high[0], high[1], high[2], high[3], low[4]])
    default = 0.5 * (low + high)
    scale = (high - low) * env_action_scale()

    for s_val, expect in ((0.0, min_v), (1.0, max_v)):
        s = torch.full((1, 5), s_val)
        a = s_to_action(s)[0]  # (5,)
        target = default + a * scale  # (5,)
        torch.testing.assert_close(target, expect, atol=1e-5, rtol=0)


def test_backward_limit_sign_is_flipped():
    """Index 4 maps in the opposite direction from the other four."""
    s = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    a = s_to_action(s)
    assert a[0, :4].sum() < 0 < a[1, :4].sum(), "four dims increase with s"
    assert a[0, 4] > 0 > a[1, 4], "backward_limit must decrease with s"


def test_step_returns_batch_not_single_env():
    """REGRESSION: ``net.step`` returns ``(batch, 5)``, not a single env.

    The stateful predictor used to return ``pred[0]``, collapsing the batch to
    one environment's parameters.  Because the environment then broadcasts that
    ``(5,)`` tensor, every environment silently shared env 0's envelope -- the
    run looked plausible while measuring only one environment's behaviour.
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.model import EnvelopeNet
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import SLConfig

    net = EnvelopeNet(SLConfig().model).eval()
    batch = 6
    obs = torch.randn(1, batch, 190)
    with torch.no_grad():
        pred, hidden = net.step(obs)
    assert pred.shape == (batch, 5), f"expected ({batch}, 5), got {tuple(pred.shape)}"
    assert hidden.shape == (1, batch, net.cfg.rnn_hidden_dim)
    # distinct inputs must give distinct predictions per env
    with torch.no_grad():
        p2, _ = net.step(torch.randn(1, batch, 190))
    assert not torch.allclose(pred, p2)


def test_window_and_stateful_agree_on_shape():
    """Both predictors must hand back one row per environment."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import _make_predictor
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.model import EnvelopeNet
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import SLConfig

    cfg = SLConfig()
    net = EnvelopeNet(cfg.model).eval()
    batch, seq = 5, cfg.train.seq_len

    stateful = _make_predictor(net, "stateful", seq, 5, "cpu")
    out = stateful(torch.randn(batch, 190))
    assert out.shape == (batch, 5), f"stateful returned {tuple(out.shape)}"

    window = _make_predictor(net, "window", seq, 5, "cpu")
    for _ in range(seq + 1):
        out = window(torch.randn(batch, 190))
    assert out.shape == (batch, 5), f"window returned {tuple(out.shape)}"


# --------------------------------------------------------------------------
# thresholds against the real closed-loop run
# --------------------------------------------------------------------------


@pytest.mark.skipif(not EVAL_JSON, reason="EA2_SL_EVAL not set")
def test_real_closed_loop_beats_baselines():
    with open(EVAL_JSON) as f:
        data = json.load(f)
    # support both the per-seed file and the merged multi-seed file
    if "results" in data and isinstance(data["results"], dict) and "sl_window" in data["results"]:
        res = data["results"]
    else:
        res = data["results"][sorted(data["results"].keys())[0]]
    for name, r in res.items():
        print(
            f"[G3] {name:<22} step_rew={r['step_reward']:+.4f} "
            f"mse={r['pred_target_mse']:.5f} collide={r['collision_rate']:.4f} "
            f"tstd={r['temporal_std']:.4f}"
        )

    # prefer the recommended deployment mode when both were evaluated
    sl = res["sl_stateful"] if "sl_stateful" in res else res["sl_window"]
    zero = res["zero_action"]
    # the strongest constant policy: the oracle target mean, estimated live.
    # Comparing against the midpoint 0.5 instead would understate the baseline
    # by ~2x (measured MSE 0.167 vs 0.083).
    const = res["constant_oracle_mean"]

    # the policy must clearly beat both trivial baselines
    assert sl["step_reward"] > zero["step_reward"], "SL policy no better than zero action"
    assert sl["step_reward"] > const["step_reward"], "SL policy no better than a constant"

    # reward model consistency: r ~= -3 * MSE (plus small action-rate/limit terms)
    implied = -3.0 * sl["pred_target_mse"]
    assert abs(sl["step_reward"] - implied) < 0.02, (
        f"reward inconsistent with MSE: {sl['step_reward']:.4f} vs implied {implied:.4f}"
    )

    # the envelope must actually change over time
    assert sl["temporal_std"] > 0.05, f"envelope too static: tstd={sl['temporal_std']:.4f}"

    # safety -- this is where the adaptive policy really earns its keep: the
    # best constant envelope minimises MSE but collides far more often, because
    # it cannot shrink for narrow passages.
    assert sl["collision_rate"] < 0.15, f"collision rate too high: {sl['collision_rate']:.4f}"
    assert sl["collision_rate"] < const["collision_rate"] * 0.6, (
        f"adaptive policy should cut collisions substantially: "
        f"{sl['collision_rate']:.4f} vs constant {const['collision_rate']:.4f}"
    )


@pytest.mark.skipif(not EVAL_JSON, reason="EA2_SL_EVAL not set")
def test_stateful_is_at_least_as_good_as_window():
    """Carrying the hidden state across the episode must not hurt.

    An earlier measurement claimed the opposite and blamed long-horizon drift;
    that was caused by the stateful predictor indexing ``pred[0]`` and so
    feeding every environment env 0's parameters.  With that fixed, the extra
    memory helps: stateful MSE is ~4x lower than window across all maps.
    """
    with open(EVAL_JSON) as f:
        res = json.load(f)["results"]
    if "sl_stateful" not in res or "sl_window" not in res:
        pytest.skip("both modes were not evaluated")
    w, s = res["sl_window"], res["sl_stateful"]
    print(f"[G3] window mse={w['pred_target_mse']:.5f}  stateful mse={s['pred_target_mse']:.5f}")
    assert s["pred_target_mse"] <= w["pred_target_mse"] * 1.1, (
        f"stateful should not be much worse than window: "
        f"{s['pred_target_mse']:.5f} vs {w['pred_target_mse']:.5f}"
    )
    assert s["step_reward"] >= w["step_reward"] - 0.02
