"""Gate G4 -- weight-transplant correctness (the riskiest step in the pipeline).

A silent partial transfer (wrong key, mismatched shape, wrong layout) would
still produce a *runnable* PPO policy -- just a useless one.  Every check here
exists to make that failure loud.
"""

from __future__ import annotations

import os

import pytest
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.sl import export as sexp
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.model import EnvelopeNet
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import SLConfig

CKPT = os.environ.get("EA2_SL_CKPT", "")


def _net_and_policy(std: float = 0.5):
    cfg = SLConfig()
    net = EnvelopeNet(cfg.model)
    policy = sexp.build_ppo_policy(cfg, device="cpu", init_noise_std=std)
    return cfg, net, policy


def test_key_map_has_ten_entries():
    net = EnvelopeNet(SLConfig().model)
    mapping = net.actor_state_dict_keys()
    assert len(mapping) == 10
    assert all(k in net.state_dict() for k in mapping)


def test_export_copies_all_actor_weights():
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, device="cpu")
    checks = sexp.verify_export(net, policy)
    for key, ok in checks.items():
        if key.startswith("equal::") or key.startswith("fold::"):
            assert ok, f"weight not transplanted correctly: {key}"


def test_export_sets_std():
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, init_noise_std=0.25, device="cpu")
    assert torch.allclose(policy.std, torch.full_like(policy.std, 0.25))
    policy2 = sexp.export(net, cfg, init_noise_std=0.5, device="cpu")
    assert torch.allclose(policy2.std, torch.full_like(policy2.std, 0.5))


def test_export_default_std_comes_from_config():
    cfg, net, _ = _net_and_policy()
    cfg.train.export_std = 0.3
    policy = sexp.export(net, cfg, device="cpu")
    assert torch.allclose(policy.std, torch.full_like(policy.std, 0.3))


def test_export_reproduces_behaviour_single_frame():
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, device="cpu")
    checks = sexp.verify_export(net, policy)
    assert checks["behaviour::single_frame"]


def test_export_folds_env_action_mapping_into_final_layer():
    """The deployed policy must realise the net's s through the env mapping.

    ``EnvelopeNet`` outputs normalised params ``s in [0, 1]`` but the env
    consumes raw actions via ``s = 0.5 + k * sign * a``.  A verbatim copy
    feeds ``s`` where ``a`` is expected and pins the envelope near its
    midpoint (realised range ``[0.5, 0.5 + k]``) -- the export therefore folds
    ``a = sign * (s - 0.5) / k`` into the final actor layer.
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import (
        ACTION_SIGN,
        env_action_scale,
    )

    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, device="cpu")
    k = env_action_scale()
    sign = torch.tensor(ACTION_SIGN)
    d = sign / k

    src = net.state_dict()
    dst = policy.state_dict()
    torch.testing.assert_close(
        dst["actor.4.weight"], src["mlp.4.weight"] * d.unsqueeze(-1)
    )
    torch.testing.assert_close(dst["actor.4.bias"], d * (src["mlp.4.bias"] - 0.5))

    # end-to-end: env-realised s equals the net output, including the endpoints
    s_vals = torch.tensor([[0.0] * 5, [0.5] * 5, [1.0] * 5, [0.2, 0.9, 0.4, 0.75, 0.6]])
    a = sign * (s_vals - 0.5) / k
    low = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9])
    high = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6])
    default = 0.5 * (low + high)
    params = torch.clamp(default + a * (high - low) * k, low, high)
    min_v = torch.stack([low[0], low[1], low[2], low[3], high[4]])
    span = torch.stack([high[0], high[1], high[2], high[3], low[4]]) - min_v
    torch.testing.assert_close((params - min_v) / span, s_vals, atol=1e-5, rtol=0)


def test_export_fold_disabled_is_verbatim_copy():
    """``fold_action_mapping=False`` restores the raw transplant (debug only)."""
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, device="cpu", fold_action_mapping=False)
    src = net.state_dict()
    dst = policy.state_dict()
    torch.testing.assert_close(dst["actor.4.weight"], src["mlp.4.weight"])
    torch.testing.assert_close(dst["actor.4.bias"], src["mlp.4.bias"])


def test_export_reproduces_behaviour_multi_step():
    """Hidden state must be carried identically across several frames."""
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, device="cpu")
    checks = sexp.verify_export(net, policy)
    assert checks["behaviour::multi_step"]


def test_export_leaves_critic_random_by_default():
    """Default behaviour: critic keeps its fresh initialisation.

    Compared in-place: two separately built policies have independent random
    initialisations, so the only meaningful check is before/after on the *same*
    instance.
    """
    cfg, net, _ = _net_and_policy()
    policy = sexp.build_ppo_policy(cfg, device="cpu")
    before = {
        k: v.clone()
        for k, v in policy.state_dict().items()
        if k.startswith("memory_c.") or k.startswith("critic.")
    }
    sexp.export(net, cfg, device="cpu", policy=policy)
    after = policy.state_dict()
    for name, tensor in before.items():
        assert torch.allclose(after[name], tensor), f"critic parameter {name} was modified"


def test_export_can_initialise_critic_on_request():
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, init_critic=True, device="cpu")
    src = net.state_dict()
    assert torch.allclose(
        policy.state_dict()["memory_c.rnn.weight_ih_l0"], src["gru.weight_ih_l0"]
    )


# --------------------------------------------------------------------------
# rsl_rl checkpoint contract -- required by train.py --resume / play_ea2.py
# --------------------------------------------------------------------------


def test_saved_checkpoint_has_all_rsl_rl_keys(tmp_path):
    """``OnPolicyRunner.load()`` dereferences four keys unconditionally.

    A checkpoint missing any of them fails at load time with KeyError, which is
    exactly how the first export attempt broke.  Guarded here so it cannot
    regress.
    """
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, device="cpu")
    path = str(tmp_path / "m0.pt")
    sexp.save_policy(policy, path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in sexp.RSL_RL_REQUIRED_KEYS if k not in raw]
    assert not missing, f"checkpoint missing rsl_rl keys: {missing}"


def test_extra_metadata_cannot_shadow_required_keys():
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, device="cpu")
    payload = sexp.build_rsl_rl_checkpoint(
        policy, learning_rate=1e-3, iteration=3, extra={"iter": 999, "infos": "bogus"}
    )
    assert payload["iter"] == 3
    assert payload["infos"] is None
    assert "model_state_dict" in payload and "optimizer_state_dict" in payload


def test_optimiser_state_matches_rsl_rl_structure():
    """The exported optimiser must have the same param_groups shape as PPO's."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import El4090EA2CfgPPO
    from rsl_rl.algorithms import PPO

    ppo_cfg = El4090EA2CfgPPO()
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, device="cpu")

    # a real PPO optimiser over the same policy
    ref_algo = PPO(
        policy=policy,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=ppo_cfg.algorithm.clip_param,
        entropy_coef=ppo_cfg.algorithm.entropy_coef,
        num_learning_epochs=ppo_cfg.algorithm.num_learning_epochs,
        num_mini_batches=ppo_cfg.algorithm.num_mini_batches,
        learning_rate=ppo_cfg.algorithm.learning_rate,
        schedule=ppo_cfg.algorithm.schedule,
        gamma=ppo_cfg.algorithm.gamma,
        lam=ppo_cfg.algorithm.lam,
        desired_kl=ppo_cfg.algorithm.desired_kl,
        max_grad_norm=ppo_cfg.algorithm.max_grad_norm,
        device="cpu",
    )
    payload = sexp.build_rsl_rl_checkpoint(policy, ppo_cfg.algorithm.learning_rate)
    # loading into the real optimiser must not raise
    ref_algo.optimizer.load_state_dict(payload["optimizer_state_dict"])
    assert ref_algo.optimizer.param_groups[0]["lr"] == ppo_cfg.algorithm.learning_rate


def test_ppo_log_path_follows_get_load_path_layout():
    """The resolved path must be the one ``helpers.get_load_path`` searches."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import El4090EA2CfgPPO

    path = sexp.ppo_log_path("myrun", iteration=0)
    exp = El4090EA2CfgPPO().runner.experiment_name
    assert path.endswith(f"logs/{exp}/myrun/model_0.pt"), path


def test_export_rejects_shape_mismatch():
    """A mismatched source must raise rather than silently skip."""
    cfg, net, _ = _net_and_policy()
    # shrink the net's GRU so its tensors no longer fit the policy
    net.gru = torch.nn.GRU(190, 32, 1, batch_first=False)
    with pytest.raises(ValueError, match="shape mismatch"):
        sexp.export(net, cfg, device="cpu")


def test_policy_can_be_saved_and_reloaded(tmp_path):
    cfg, net, _ = _net_and_policy()
    policy = sexp.export(net, cfg, device="cpu")
    path = str(tmp_path / "policy.pt")
    sexp.save_policy(policy, path, extra={"source": "sl"})
    raw = torch.load(path)
    assert "model_state_dict" in raw
    fresh = sexp.build_ppo_policy(cfg, device="cpu")
    fresh.load_state_dict(raw["model_state_dict"], strict=True)


# --------------------------------------------------------------------------
# against the real trained checkpoint
# --------------------------------------------------------------------------


@pytest.mark.skipif(not CKPT, reason="EA2_SL_CKPT not set")
def test_real_checkpoint_exports_cleanly():
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import load_checkpoint

    net, meta = load_checkpoint(CKPT, device="cpu")
    cfg = SLConfig()
    fresh = sexp.build_ppo_policy(cfg, device="cpu")
    reference = {k: v.detach().cpu().clone() for k, v in fresh.state_dict().items()}
    policy = sexp.export(net, cfg, device="cpu", policy=fresh)
    checks = sexp.verify_export(net, policy, reference_state=reference)
    bad = [k for k, v in checks.items() if not v]
    assert not bad, f"export verification failed for: {bad}"
    print(f"[G4] real checkpoint exported, best_epoch={meta.get('best_epoch')}")
    print(f"[G4] val_r2={meta.get('val', {}).get('r2'):.4f}  std={float(policy.std.flatten()[0]):.2f}")
