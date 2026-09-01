"""Cascade contract tests (no Isaac env construction).

Covers the pure-torch side of the EA2→SE2 merge: the envelope bridge math,
its idempotency with the SE2 condition state, and the EA2 policy loader.
Run via the repo's ea2 test suite (``python -m pytest legged_gym/tests/ea2 -q``)
or directly: ``python legged_gym/tests/ea2/cascade/test_cascade_contracts.py``.
"""

import isaacgym  # noqa: F401  -- must precede legged_gym imports

from pathlib import Path

import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
    El4090EA2Cfg,
)
from legged_gym.envs.el_4090.envelope_cascade_83.ea2_policy import Ea2Policy
from legged_gym.envs.el_4090.envelope_cascade_83.envelope_bridge import EnvelopeBridge
from legged_gym.envs.el_4090.spider_envelop_2.el4090_spider_config import (
    El4090Envelop2Cfg,
)
from legged_gym.envs.el_4090.spider_envelop_2.envelope_condition import (
    EnvelopeConditionState,
)
from legged_gym.utils.envelop.network.haa_swing_range import (
    apply_env_morphology_priors,
    load_envelope_condition_spec,
)
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c

_CASCADE_DIR = Path(__file__).resolve().parents[3] / "envs" / "el_4090" / "envelope_cascade_83"
_PINNED_CKPT = _CASCADE_DIR / "checkpoints" / "ea2_envelope.pt"
_LOGS_CKPT = Path(__file__).resolve().parents[4] / "logs" / "el4090_ea2" / "v2_multik" / "model_0.pt"


def _make_envelope_state(num_envs: int = 3) -> EnvelopeConditionState:
    env_cfg = El4090Envelop2Cfg.envelope

    class _Ranges:
        pass

    class _FakeEnvCfg:
        pass

    fake = _FakeEnvCfg()
    fake.condition_names = list(env_cfg.condition_names)
    fake.morphology_prior_mode = env_cfg.morphology_prior_mode
    fake.morphology_prior_weights = env_cfg.morphology_prior_weights
    fake.morphology_middle_front_follow_weight = (
        env_cfg.morphology_middle_front_follow_weight
    )
    fake.ranges = _Ranges()
    for name in fake.condition_names:
        setattr(fake.ranges, name, list(getattr(env_cfg.ranges, name)))
    return EnvelopeConditionState(fake, num_envs=num_envs, device="cpu")


def _find_checkpoint():
    if _PINNED_CKPT.exists():
        return _PINNED_CKPT
    if _LOGS_CKPT.exists():
        return _LOGS_CKPT
    return None


# ── bridge math ──────────────────────────────────────────────────────────


def test_bridge_fold_scale_matches_live_ea2_config():
    state = _make_envelope_state()
    soft = float(El4090EA2Cfg.envelope.soft_dof_pos_limit)
    action_max = float(El4090EA2Cfg.envelope.action_max)
    bridge = EnvelopeBridge(state, soft, action_max)
    assert abs(bridge.fold_scale - 0.11875) < 1e-9


def test_bridge_maps_raw_action_like_ea2_env():
    state = _make_envelope_state()
    bridge = EnvelopeBridge(
        state,
        float(El4090EA2Cfg.envelope.soft_dof_pos_limit),
        float(El4090EA2Cfg.envelope.action_max),
    )
    raw = torch.tensor([[0.0, 4.0, -4.0, 2.0, -2.0], [1.0, -1.0, 0.5, 8.0, -8.0]])
    params5 = bridge.params_from_action(raw)
    mid = 0.5 * (bridge.low5 + bridge.high5)
    expected = torch.clamp(
        mid + raw * (bridge.high5 - bridge.low5) * bridge.fold_scale,
        bridge.low5,
        bridge.high5,
    )
    assert torch.allclose(params5, expected, atol=1e-6)
    # saturation: outward raw actions reach the hard bounds on every column
    # (backward_limit: more negative = larger rear extent, so a=-8 → low)
    saturated = bridge.params_from_action(torch.tensor([[8.0, 8.0, 8.0, 8.0, -8.0]]))
    expected_sat = torch.cat(
        [bridge.high5[:4], bridge.low5[4:5]], dim=0
    ).unsqueeze(0)
    assert torch.allclose(saturated, expected_sat, atol=1e-6)
    zero = bridge.params_from_action(torch.zeros(1, 5))
    assert torch.allclose(zero, mid[None], atol=1e-6)


def test_bridge_condition_is_idempotent_and_in_bounds():
    state = _make_envelope_state()
    spec = load_envelope_condition_spec(ea2c.ENVELOPE_SPEC_CONFIG_PATH)
    bridge = EnvelopeBridge(
        state,
        float(El4090EA2Cfg.envelope.soft_dof_pos_limit),
        float(El4090EA2Cfg.envelope.action_max),
    )
    raw = torch.randn(4, 5) * 3.0
    cond8, params5 = bridge(raw)
    assert cond8.shape == (4, 8)
    assert torch.allclose(cond8[..., :5], params5)
    low = torch.tensor(spec.low)
    high = torch.tensor(spec.high)
    assert bool(((cond8 >= low) & (cond8 <= high)).all())
    # applying priors again must not change anything (idempotent derivation)
    twice = apply_env_morphology_priors(cond8.clone(), spec)
    assert torch.allclose(cond8, twice, atol=1e-6)


def test_envelope_state_accepts_bridge_output():
    state = _make_envelope_state()
    bridge = EnvelopeBridge(
        state,
        float(El4090EA2Cfg.envelope.soft_dof_pos_limit),
        float(El4090EA2Cfg.envelope.action_max),
    )
    cond8, _ = bridge(torch.randn(3, 5) * 4.0)
    updated = state.set(cond8, derive_priors=True)
    assert torch.allclose(state.get(), cond8, atol=1e-6)
    assert torch.allclose(updated, cond8, atol=1e-6)
    # out-of-range geometry columns are clamped by the state
    wild = cond8.clone()
    wild[:, 0] = 99.0
    state.set(wild, derive_priors=True)
    assert torch.allclose(state.get()[:, 0], state.high[0].expand(3))


# ── policy loader ────────────────────────────────────────────────────────


def test_policy_loader_rejects_missing_checkpoint():
    import pytest

    with pytest.raises(FileNotFoundError):
        Ea2Policy(Path("/nonexistent/ea2_envelope.pt"), device="cpu")


def test_policy_loads_real_checkpoint_and_runs():
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        print("SKIP: no EA2 checkpoint pinned or in logs yet")
        return
    policy = Ea2Policy(ckpt_path, device="cpu")
    assert policy.num_observations == 190
    assert policy.num_actions == 5
    obs = torch.rand(2, 190)
    action = policy.act(obs)
    assert action.shape == (2, 5)
    assert bool(torch.isfinite(action).all())
    policy.reset(torch.tensor([True, False]))
    action2 = policy.act(torch.rand(2, 190))
    assert action2.shape == (2, 5)


def test_policy_loader_rejects_wrong_obs_dim():
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        print("SKIP: no EA2 checkpoint pinned or in logs yet")
        return
    import pytest

    policy = Ea2Policy(ckpt_path, device="cpu")
    with pytest.raises(ValueError):
        policy.act(torch.rand(2, 83))


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"===== failures: {failures} =====")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_all())
