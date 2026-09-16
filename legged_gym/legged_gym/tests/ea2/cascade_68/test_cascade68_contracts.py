"""Cascade_68 contract tests (no Isaac env construction).

Covers the pure-torch side of the EA2→SE2 merge: the envelope bridge math,
its idempotency with the SE2 condition state, the EA2 policy loader, and the
frozen 68-dim contract pins (weight pointers inside the package, pinned
checkpoint checksums).

Structural twin of ``tests/ea2/cascade/test_cascade_contracts.py`` (the
legacy 83-dim task); the two suites are independent.
Run via the repo's ea2 test suite (``python -m pytest legged_gym/tests/ea2 -q``)
or directly: ``python legged_gym/tests/ea2/cascade_68/test_cascade68_contracts.py``.
"""

import isaacgym  # noqa: F401  -- must precede legged_gym imports

from pathlib import Path

import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
    El4090EA2Cfg,
)
from legged_gym.envs.el_4090.envelope_cascade_68.ea2_policy import Ea2Policy
from legged_gym.envs.el_4090.envelope_cascade_68.el4090_cascade_config import (
    El4090Cascade68Cfg,
)
from legged_gym.envs.el_4090.envelope_cascade_68.envelope_bridge import EnvelopeBridge
from legged_gym.envs.el_4090.envelope_cascade_68.se2_frozen.config import (
    El4090Se2_68Cfg,
)
from legged_gym.envs.el_4090.envelope_cascade_68.se2_frozen.envelope_condition import (
    EnvelopeConditionState,
)
from legged_gym.utils.envelop.network.haa_swing_range import (
    apply_env_morphology_priors,
    load_envelope_condition_spec,
)
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c

_CASCADE_DIR = Path(__file__).resolve().parents[3] / "envs" / "el_4090" / "envelope_cascade_68"
_PINNED_CKPT = _CASCADE_DIR / "checkpoints" / "ea2_envelope_v3att.pt"
_LOGS_CKPT = Path(__file__).resolve().parents[4] / "logs" / "el4090_ea2" / "v3_attitude" / "model_0.pt"

_MD5_HAA_RANGE = "640627fd6a831cfbfcc308b5772101cf"
_MD5_EA2_V3ATT = "d294151ce917180fbbb06cc6aea6c3e1"
_MD5_EA2_V2_BACKUP = "4716b023632a5712265fcc672e272306"
# Set together with placing checkpoints/policy_1.pt (see checkpoints/README.md).
# Until then the SE2-policy md5 test skips; a placed file with this still None
# is a loud FAIL (weights must never land without their contract pin).
_PINNED_SE2_POLICY_MD5 = None


def _make_envelope_state(num_envs: int = 3) -> EnvelopeConditionState:
    env_cfg = El4090Se2_68Cfg.envelope

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


def _md5(path: Path) -> str:
    import hashlib

    return hashlib.md5(path.read_bytes()).hexdigest()


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
        policy.act(torch.rand(2, 68))


# ── frozen 68-dim contract ───────────────────────────────────────────────


def test_cascade_68_frozen_observation_contract():
    """cascade_68 pins the CURRENT 68-dim SE2 contract (vendored from
    spider_envelop_2 at 84c08ca, the state after the range priors were
    removed from the policy observation).  Do NOT align this task with any
    future upstream observation change: policy_1.pt will be a 68-dim
    TorchScript and cannot consume any other layout.  The legacy 83-dim
    contract lives in envelope_cascade_83; the two are not interchangeable.
    """
    env = El4090Cascade68Cfg.env
    assert env.num_observations == 68
    assert env.num_physical_priors == 0
    assert env.num_haa_range_observations == 0
    scales = El4090Cascade68Cfg.normalization.obs_scales
    # the three prior-era obs_scales keys must stay absent (a re-addition
    # would mean the observation layout drifted back toward the 83-dim form)
    assert getattr(scales, "morphology_prior", None) is None
    assert getattr(scales, "haa_range_center", None) is None
    assert getattr(scales, "haa_range_half", None) is None


def test_cascade_68_weight_pointers_stay_inside_the_package():
    """All three weight pointers must resolve into this package's
    checkpoints/ (the inherited se2_frozen defaults name external paths and
    MUST stay overridden — a silent revert would desync the pinned set).
    """
    marker = "envelope_cascade_68/checkpoints"
    assert marker in El4090Cascade68Cfg.haa_swing_range.network_checkpoint
    assert marker in El4090Cascade68Cfg.se2_policy.checkpoint
    assert marker in El4090Cascade68Cfg.ea2.checkpoint
    assert "envelope_adaptive_2/selected_airy_channels.pt" in (
        El4090Cascade68Cfg.ea2.channel_file
    )


def test_pinned_haa_and_ea2_checkpoints_unchanged():
    """Byte-level freeze of the pinned weights (update together with
    checkpoints/README.md on an intentional re-pin)."""
    haa = _CASCADE_DIR / "checkpoints" / "haa_range.pt"
    ea2_v3 = _CASCADE_DIR / "checkpoints" / "ea2_envelope_v3att.pt"
    ea2_v2 = _CASCADE_DIR / "checkpoints" / "ea2_envelope.pt"
    assert haa.exists(), f"missing pinned HAA network: {haa}"
    assert ea2_v3.exists(), f"missing pinned EA2 policy: {ea2_v3}"
    assert ea2_v2.exists(), f"missing rollback EA2 policy: {ea2_v2}"
    assert _md5(haa) == _MD5_HAA_RANGE, f"haa_range.pt changed (md5 {_md5(haa)})"
    assert _md5(ea2_v3) == _MD5_EA2_V3ATT, (
        f"ea2_envelope_v3att.pt changed (md5 {_md5(ea2_v3)})"
    )
    assert _md5(ea2_v2) == _MD5_EA2_V2_BACKUP, (
        f"ea2_envelope.pt changed (md5 {_md5(ea2_v2)})"
    )


def test_pinned_se2_policy_checkpoint_unchanged():
    """policy_1.pt (68→18 TorchScript) is reserved but NOT yet placed (68-dim
    SE2 retrain pending — see checkpoints/README.md).  Once placed, this test
    hard-pins its md5; a file without its pin is a loud failure."""
    ckpt = _CASCADE_DIR / "checkpoints" / "policy_1.pt"
    if not ckpt.exists():
        print("SKIP: SE2 68-dim policy not pinned yet (see checkpoints/README.md)")
        return
    assert _PINNED_SE2_POLICY_MD5, (
        "policy_1.pt exists but _PINNED_SE2_POLICY_MD5 is unset — update "
        "the pin here and checkpoints/README.md together"
    )
    md5 = _md5(ckpt)
    assert md5 == _PINNED_SE2_POLICY_MD5, (
        f"policy_1.pt changed (md5 {md5}); if retrained intentionally, update "
        "this pin and checkpoints/README.md together"
    )


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
