"""Gate G1 -- model-structure checks for the EA2 supervised-learning policy.

The central requirement is *export parity*: ``EnvelopeNet`` must be parameter-for-
parameter compatible with rsl_rl's ``ActorCriticRecurrent`` actor half, so that
block B4 can transplant weights without any reshaping.
"""

from __future__ import annotations

import pytest
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.sl.model import EnvelopeNet
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import SLModelConfig

# rsl_rl expects these defaults from el_4090_ea2_config.policy
RNN_HIDDEN = 187
OBS_DIM = 190
ACTION_DIM = 5
HIDDEN_DIMS = [256, 128]


def _net() -> EnvelopeNet:
    return EnvelopeNet(SLModelConfig())


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


def test_forward_shape():
    net = _net()
    seq, batch = 40, 8
    obs = torch.randn(seq, batch, OBS_DIM)
    out = net(obs)
    assert out.shape == (seq, batch, ACTION_DIM)


def test_step_shape():
    """step() consumes one frame and returns (batch, 5) plus hidden state."""
    net = _net()
    batch = 6
    obs = torch.randn(1, batch, OBS_DIM)
    pred, hidden = net.step(obs)
    assert pred.shape == (batch, ACTION_DIM)
    assert hidden.shape == (1, batch, RNN_HIDDEN)


def test_step_rejects_wrong_layout():
    net = _net()
    with pytest.raises(ValueError):
        net.step(torch.randn(4, 6, OBS_DIM))  # missing leading time axis


def test_step_matches_forward_last_frame():
    """step-by-step recurrence must equal the full-sequence forward pass."""
    net = _net().eval()
    seq, batch = 12, 4
    obs = torch.randn(seq, batch, OBS_DIM)
    with torch.no_grad():
        full = net(obs)
        hidden = None
        outs = []
        for t in range(seq):
            p, hidden = net.step(obs[t : t + 1], hidden)
            outs.append(p)
        stepped = torch.stack(outs)
    torch.testing.assert_close(stepped, full, atol=1e-5, rtol=1e-5)


def test_hidden_state_defaults_to_zero():
    net = _net().eval()
    obs = torch.randn(1, 3, OBS_DIM)
    with torch.no_grad():
        p_default, _ = net.step(obs)
        p_zero, _ = net.step(obs, net.init_hidden(3))
    torch.testing.assert_close(p_default, p_zero)


# --------------------------------------------------------------------------
# optimisability
# --------------------------------------------------------------------------


def test_gradient_reaches_gru():
    net = _net()
    obs = torch.randn(10, 4, OBS_DIM)
    target = torch.rand(10, 4, ACTION_DIM)
    loss = torch.nn.functional.mse_loss(net(obs), target)
    loss.backward()
    assert net.gru.weight_ih_l0.grad is not None
    assert torch.isfinite(net.gru.weight_ih_l0.grad).all()
    assert float(net.gru.weight_ih_l0.grad.abs().sum()) > 0
    assert net.mlp[4].weight.grad is not None


def test_overfits_tiny_batch():
    """Sanity: the net must be able to memorise a handful of sequences."""
    torch.manual_seed(0)
    net = _net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    obs = torch.randn(8, 4, OBS_DIM)
    tgt = torch.rand(8, 4, ACTION_DIM)
    for _ in range(200):
        loss = torch.nn.functional.mse_loss(net(obs), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.01, f"failed to overfit, final loss {float(loss):.4f}"


# --------------------------------------------------------------------------
# export parity (the reason this file exists)
# --------------------------------------------------------------------------


def _rsl_rl_policy():
    from rsl_rl.modules import ActorCriticRecurrent

    return ActorCriticRecurrent(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=ACTION_DIM,
        actor_hidden_dims=list(HIDDEN_DIMS),
        critic_hidden_dims=list(HIDDEN_DIMS),
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=RNN_HIDDEN,
        rnn_num_layers=1,
        init_noise_std=0.5,
    )


def test_export_key_map_covers_all_actor_params():
    net = _net()
    mapping = net.actor_state_dict_keys()
    src = net.state_dict()
    assert set(mapping.keys()) == set(src.keys()), "every net parameter must be mapped"
    assert len(mapping) == 10


@pytest.mark.parametrize("key", [
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
    "mlp.4.weight", "mlp.4.bias",
])
def test_export_shapes_match_rsl_rl(key):
    """Each mapped tensor must have exactly the rsl_rl target shape."""
    net = _net()
    policy = _rsl_rl_policy()
    mapping = net.actor_state_dict_keys()
    src_t = net.state_dict()[key]
    dst_t = policy.state_dict()[mapping[key]]
    assert src_t.shape == dst_t.shape, f"{key}: {tuple(src_t.shape)} vs {tuple(dst_t.shape)}"


def test_param_count_matches_actor_half():
    """Net params == rsl_rl (GRU a + actor MLP), i.e. exactly the actor half."""
    net = _net()
    policy = _rsl_rl_policy()
    gru_params = sum(p.numel() for p in policy.memory_a.parameters())
    actor_mlp = sum(p.numel() for p in policy.actor.parameters())
    assert net.param_count() == gru_params + actor_mlp
    # documented figures: GRU 212,619 + MLP 81,669
    assert net.param_count() == 294_288


def test_transplant_then_behaviour_identical():
    """After copying weights, the rsl_rl policy must reproduce net.forward().

    Note the calling convention difference: rsl_rl's ``Memory`` in inference
    mode takes a 2-D ``(batch, obs)`` and does ``input.unsqueeze(0)`` itself
    (see ``rsl_rl/networks/memory.py:32``), whereas ``EnvelopeNet`` takes the
    explicit ``(seq, batch, obs)`` layout.  The weights are identical; only the
    tensor rank at the call site differs.
    """
    net = _net().eval()
    policy = _rsl_rl_policy().eval()
    sd = policy.state_dict()
    net_sd = net.state_dict()
    for src_key, dst_key in net.actor_state_dict_keys().items():
        sd[dst_key].copy_(net_sd[src_key])
    policy.load_state_dict(sd, strict=True)

    batch = 5
    obs_2d = torch.randn(batch, OBS_DIM)
    with torch.no_grad():
        net_out = net(obs_2d.unsqueeze(0))  # (1, batch, obs) -> (1, batch, 5)
        # inference mode: 2-D input, Memory unsqueezes internally
        mem_out = policy.memory_a(obs_2d)  # -> (1, batch, hidden)
        pol_out = policy.actor(mem_out)  # -> (1, batch, 5)
    torch.testing.assert_close(pol_out, net_out, atol=1e-6, rtol=1e-6)


def test_multi_step_transplant_matches_step_api():
    """A short rollout must agree frame-by-frame between the two call styles."""
    net = _net().eval()
    policy = _rsl_rl_policy().eval()
    sd = policy.state_dict()
    net_sd = net.state_dict()
    for src_key, dst_key in net.actor_state_dict_keys().items():
        sd[dst_key].copy_(net_sd[src_key])
    policy.load_state_dict(sd, strict=True)

    seq, batch = 6, 3
    obs = torch.randn(seq, batch, OBS_DIM)
    with torch.no_grad():
        net_hidden = None
        net_outs = []
        for t in range(seq):
            p, net_hidden = net.step(obs[t : t + 1], net_hidden)
            net_outs.append(p)

        policy.memory_a.hidden_states = None  # start from zero
        pol_outs = []
        for t in range(seq):
            mem_out = policy.memory_a(obs[t])  # (batch, obs)
            pol_outs.append(policy.actor(mem_out)[0])  # (batch, 5)
    torch.testing.assert_close(torch.stack(pol_outs), torch.stack(net_outs), atol=1e-6, rtol=1e-6)
