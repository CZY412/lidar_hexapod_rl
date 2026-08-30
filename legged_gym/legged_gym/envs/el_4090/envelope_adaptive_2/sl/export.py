"""Export a trained ``EnvelopeNet`` into an rsl_rl ``ActorCriticRecurrent`` (B4).

The two networks are isomorphic by construction (see ``model.py``), so the
transplant is a straight key-by-key copy of ten tensors:

    gru.weight_ih_l0 -> memory_a.rnn.weight_ih_l0
    gru.weight_hh_l0 -> memory_a.rnn.weight_hh_l0
    gru.bias_ih_l0   -> memory_a.rnn.bias_ih_l0
    gru.bias_hh_l0   -> memory_a.rnn.bias_hh_l0
    mlp.0.weight     -> actor.0.weight
    mlp.0.bias       -> actor.0.bias
    mlp.2.weight     -> actor.2.weight
    mlp.2.bias       -> actor.2.bias
    mlp.4.weight     -> actor.4.weight
    mlp.4.bias       -> actor.4.bias

plus ``std``, which has no counterpart in the SL net and is set explicitly.

What is *not* transferred
-------------------------
The critic side (``memory_c`` + ``critic``) stays randomly initialised -- the SL
net never had one.  An option exists to copy the actor's GRU into the critic's
(``init_critic=True``) so the value function starts from a meaningful geometry
encoder; this is **off by default** because mixing a pretrained encoder into a
randomly-initialised value head can destabilise early PPO updates.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .model import EnvelopeNet
from .sl_config import SLConfig


def build_ppo_policy(cfg: SLConfig, device: str = "cpu", init_noise_std: Optional[float] = None):
    """Instantiate an ``ActorCriticRecurrent`` matching the EA2 PPO config."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import El4090EA2Cfg, El4090EA2CfgPPO
    from rsl_rl.modules import ActorCriticRecurrent

    env_cfg, ppo_cfg = El4090EA2Cfg(), El4090EA2CfgPPO()
    return ActorCriticRecurrent(
        num_actor_obs=int(env_cfg.env.num_observations),
        num_critic_obs=int(env_cfg.env.num_observations),
        num_actions=int(env_cfg.env.num_actions),
        actor_hidden_dims=list(ppo_cfg.policy.actor_hidden_dims),
        critic_hidden_dims=list(ppo_cfg.policy.critic_hidden_dims),
        activation=ppo_cfg.policy.activation,
        rnn_type=ppo_cfg.policy.rnn_type,
        rnn_hidden_dim=ppo_cfg.policy.rnn_hidden_dim,
        rnn_num_layers=ppo_cfg.policy.rnn_num_layers,
        init_noise_std=float(init_noise_std if init_noise_std is not None else cfg.train.export_std),
    ).to(device)


def export(
    net: EnvelopeNet,
    cfg: SLConfig,
    init_noise_std: Optional[float] = None,
    init_critic: bool = False,
    device: str = "cpu",
    policy=None,
):
    """Transplant ``net``'s weights into a PPO policy.

    Args:
        policy: an existing ``ActorCriticRecurrent`` to fill **in place**.  When
            omitted a fresh one is created.  Passing one in lets callers (and
            tests) verify that untouched parameters really are untouched.

    Returns the policy; raises on any shape mismatch so that a silent partial
    transfer is impossible.
    """
    if policy is None:
        policy = build_ppo_policy(cfg, device=device, init_noise_std=init_noise_std)
    src = net.state_dict()
    dst = policy.state_dict()

    for src_key, dst_key in net.actor_state_dict_keys().items():
        if dst_key not in dst:
            raise KeyError(f"policy has no parameter {dst_key!r}")
        if dst[dst_key].shape != src[src_key].shape:
            raise ValueError(
                f"shape mismatch {src_key} -> {dst_key}: "
                f"{tuple(src[src_key].shape)} vs {tuple(dst[dst_key].shape)}"
            )
        dst[dst_key].copy_(src[src_key].to(dst[dst_key].device))

    std_val = float(init_noise_std if init_noise_std is not None else cfg.train.export_std)
    dst["std"].copy_(torch.full_like(dst["std"], std_val))

    if init_critic:
        # reuse the actor's encoder so the value head starts from geometry features
        for src_key, dst_key in net.actor_state_dict_keys().items():
            crit_key = dst_key.replace("memory_a.", "memory_c.").replace("actor.", "critic.")
            if crit_key in dst and dst[crit_key].shape == src[src_key].shape:
                dst[crit_key].copy_(src[src_key].to(dst[crit_key].device))

    # strict=True also guards against rsl_rl adding/removing parameters
    policy.load_state_dict(dst, strict=True)
    return policy


def verify_export(
    net: EnvelopeNet, policy, atol: float = 1e-6, reference_state: Optional[Dict] = None
) -> Dict[str, bool]:
    """Assert the transplanted policy reproduces the net's predictions.

    rsl_rl's ``Memory`` takes a 2-D ``(batch, obs)`` in inference mode and adds
    the time axis itself, whereas ``EnvelopeNet`` expects ``(seq, batch, obs)``.
    Both are exercised here.

    Args:
        reference_state: a snapshot of the policy's state dict taken *before*
            the transplant.  When supplied, the critic-side parameters are
            checked against it for real, instead of being assumed untouched.
    """
    net.eval()
    policy.eval()
    checks: Dict[str, bool] = {}

    src = net.state_dict()
    dst = policy.state_dict()
    for src_key, dst_key in net.actor_state_dict_keys().items():
        checks[f"equal::{dst_key}"] = bool(
            torch.allclose(src[src_key].cpu(), dst[dst_key].cpu(), atol=atol)
        )

    batch = 7
    obs2d = torch.randn(batch, net.cfg.obs_dim)
    with torch.no_grad():
        net_out = net(obs2d.unsqueeze(0))  # (1, batch, 5)
        pol_out = policy.actor(policy.memory_a(obs2d))  # (1, batch, 5)
    checks["behaviour::single_frame"] = bool(
        torch.allclose(net_out.cpu(), pol_out.cpu(), atol=1e-5)
    )

    # multi-step: hidden state must be carried identically
    seq = 5
    obs_seq = torch.randn(seq, batch, net.cfg.obs_dim)
    with torch.no_grad():
        net_hidden = None
        net_outs = []
        for t in range(seq):
            p, net_hidden = net.step(obs_seq[t : t + 1], net_hidden)
            net_outs.append(p)
        policy.memory_a.hidden_states = None
        pol_outs = []
        for t in range(seq):
            pol_outs.append(policy.actor(policy.memory_a(obs_seq[t]))[0])
    checks["behaviour::multi_step"] = bool(
        torch.allclose(
            torch.stack(net_outs).cpu(), torch.stack(pol_outs).cpu(), atol=1e-5
        )
    )

    if reference_state is not None:
        current = policy.state_dict()
        critic_keys = [
            k for k in current if k.startswith("memory_c.") or k.startswith("critic.")
        ]
        checks["critic_untouched"] = all(
            bool(torch.allclose(current[k].cpu(), reference_state[k].cpu()))
            for k in critic_keys
        )
        # the actor side must, by contrast, have changed
        checks["actor_written"] = not all(
            bool(torch.allclose(current[k].cpu(), reference_state[k].cpu()))
            for k in ("memory_a.rnn.weight_ih_l0", "actor.4.weight")
        )
    return checks


# Keys that rsl_rl's OnPolicyRunner.load() dereferences unconditionally
# (see rsl_rl/rsl_rl/runners/on_policy_runner.py).  A checkpoint missing any
# of them raises KeyError during `train.py --resume` or `play_ea2.py`.
RSL_RL_REQUIRED_KEYS = ("model_state_dict", "optimizer_state_dict", "iter", "infos")


def build_rsl_rl_checkpoint(
    policy, learning_rate: float, iteration: int = 0, extra: Optional[Dict] = None
) -> Dict:
    """Assemble a checkpoint in the exact shape rsl_rl expects.

    ``optimizer_state_dict`` must be present even though a freshly exported
    policy has no optimiser history: ``load()`` reads it whenever
    ``load_optimizer`` is true (the default).  Building it from an Adam over the
    same parameters keeps the ``param_groups`` structure identical, and using
    the PPO learning rate means fine-tuning starts at the intended step size.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    payload = dict(extra or {})
    # required keys last so that `extra` cannot shadow them
    payload.update(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": int(iteration),
            "infos": None,
        }
    )
    return payload


def save_policy(
    policy,
    path: str,
    extra: Optional[Dict] = None,
    learning_rate: Optional[float] = None,
    iteration: int = 0,
) -> None:
    """Save a policy in rsl_rl's checkpoint format.

    The result is loadable by ``OnPolicyRunner.load()``, hence by
    ``scripts/train.py --resume`` and ``scripts/play_ea2.py``.
    """
    import os

    if learning_rate is None:
        from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import El4090EA2CfgPPO

        learning_rate = float(El4090EA2CfgPPO().algorithm.learning_rate)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(
        build_rsl_rl_checkpoint(policy, learning_rate, iteration, extra), path
    )


def ppo_log_path(run_name: str, experiment_name: Optional[str] = None, iteration: int = 0) -> str:
    """Resolve ``logs/<experiment>/<run>/model_<iter>.pt``.

    This is the layout ``get_load_path`` searches, so writing there makes the
    checkpoint immediately reachable via ``--resume --load_run <run>``.
    """
    import os

    from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import El4090EA2CfgPPO
    from legged_gym.utils.helpers import LEGGED_GYM_ROOT_DIR

    if experiment_name is None:
        experiment_name = El4090EA2CfgPPO().runner.experiment_name
    return os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", experiment_name, run_name, f"model_{int(iteration)}.pt"
    )
