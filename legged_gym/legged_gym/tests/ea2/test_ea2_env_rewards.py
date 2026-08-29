"""End-to-end unit test for ``EL_4090_EA2._compute_rewards`` wiring.

Builds a bare instance via ``object.__new__`` (no sim), feeds synthetic
buffers on a synthetic corridor distance field, and asserts that the reward
buffer equals the scale-weighted sum of the documented terms and that the
episode metrics/sums accumulate the documented quantities.  Guards the
reward composition against silent drift; the individual term helpers have
their own unit tests.
"""

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import numpy as np
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
    El4090EA2Cfg,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    EL_4090_EA2,
    envelope_limit_violation,
    normalized_envelope_params,
    potential_reward,
    raw_action_rate_term,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    hex_collision_terms,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
    compute_direct_oracle_params_with_stats,
)

try:  # pytest: package ``ea2``
    from . import _ea2_testlib as tl
except ImportError:  # direct script execution
    import _ea2_testlib as tl


def _bare_env(device="cpu"):
    env = object.__new__(EL_4090_EA2)
    cfg = El4090EA2Cfg()
    env.cfg = cfg
    env.device = device
    n = 4
    env.num_envs = n
    low = tl.LOW.clone()
    high = tl.HIGH.clone()
    env._envelope_low_dev = low
    env._envelope_high_dev = high
    env.reward_scales = {
        "potential": 1.0,
        "collision": 2.0,
        "action_rate": 0.5,
        "envelope_limits": 1.0,
        "oracle_mse": 3.0,
    }
    env.rew_buf = torch.zeros(n, device=device)
    env.episode_sums = {k: torch.zeros(n, device=device) for k in env.reward_scales}
    env.episode_metrics = {
        "collision_hard_max": torch.zeros(n, device=device),
        "raw_action_abs_mean": torch.zeros(n, device=device),
        "envelope_limits_active_ratio": torch.zeros(n, device=device),
        "oracle_unsafe_before_ratio": torch.zeros(n, device=device),
        "oracle_unsafe_ratio": torch.zeros(n, device=device),
        "oracle_potential": torch.zeros(n, device=device),
        **{
            f"oracle_mse_{part}": torch.zeros(n, device=device)
            for part in (
                "front_width", "middle_width", "back_width",
                "forward_limit", "backward_limit",
            )
        },
    }
    df, _ = tl.corridor_field(1.0)
    env.distance_field = torch.as_tensor(df, dtype=torch.float32, device=device)
    env.heading = torch.tensor([0.0, 0.5, 1.0, -0.7], device=device)
    env.base_pos = torch.tensor(
        [
            [0.0, 0.0, 0.52],
            [1.0, 0.2, 0.52],
            [2.0, -0.3, 0.52],
            [-1.5, 0.1, 0.52],
        ],
        device=device,
    )
    # mixed envelopes: env 0/2 wide-open, env 1/3 narrow
    env.actions_mapped = torch.stack([tl.HIGH, tl.LOW, tl.HIGH, tl.LOW]).to(device)
    env.actions = torch.tensor(
        [[2.0, 0.0, -1.0, 0.5, 1.0]] * 2 + [[-2.0, 1.0, 0.0, -0.5, -1.0]] * 2,
        device=device,
    )
    env.last_actions_raw = env.actions * 0.5
    env.actions_target = env.actions * 0.1
    return env


def test_compute_rewards_wiring_and_metrics():
    env = _bare_env()
    low, high = env._envelope_low_dev, env._envelope_high_dev
    df = env.distance_field
    pos2 = env.base_pos[:, :2]
    margin = float(env.cfg.envelope.margin)
    soft = float(env.cfg.envelope.soft_margin)
    interp = bool(getattr(env.cfg.envelope, "oracle_interp_crossing", True))

    env._compute_rewards()

    # expected terms, computed from the same documented helpers
    potential = potential_reward(env.actions_mapped, low, high)
    violation, hard = hex_collision_terms(
        env.actions_mapped, env.heading, pos2, df, margin=margin, soft_margin=soft
    )
    act_rate = raw_action_rate_term(env.actions, env.last_actions_raw)
    limits = envelope_limit_violation(
        env.actions_target, low, high, float(env.cfg.envelope.soft_dof_pos_limit)
    )
    oracle, oracle_hard_raw = compute_direct_oracle_params_with_stats(
        env.heading, pos2, df, low, high,
        margin=float(env.cfg.envelope.oracle_margin),
        step=float(env.cfg.envelope.oracle_step),
        max_dist=float(env.cfg.envelope.oracle_max_dist),
        soft_dof_pos_limit=float(env.cfg.envelope.soft_dof_pos_limit),
        interp_crossing=interp,
    )
    oracle_mse = (
        (normalized_envelope_params(env.actions_mapped, low, high)
         - normalized_envelope_params(oracle, low, high)) ** 2
    ).mean(dim=-1)

    expected = (
        potential * env.reward_scales["potential"]
        + violation * env.reward_scales["collision"]
        + act_rate * env.reward_scales["action_rate"]
        + limits * env.reward_scales["envelope_limits"]
        + oracle_mse * env.reward_scales["oracle_mse"]
    )
    assert torch.allclose(env.rew_buf, expected, atol=1e-6)
    for name, term in (
        ("potential", potential),
        ("collision", violation),
        ("action_rate", act_rate),
        ("envelope_limits", limits),
        ("oracle_mse", oracle_mse),
    ):
        assert torch.allclose(
            env.episode_sums[name], term * env.reward_scales[name], atol=1e-6
        )

    # oracle hard collision re-check on the returned target
    _, oracle_hard = hex_collision_terms(
        oracle, env.heading, pos2, df, margin=margin, soft_margin=soft
    )
    assert torch.allclose(env._collision_hard, hard)
    assert torch.allclose(
        env.episode_metrics["collision_hard_max"], hard
    )
    assert torch.allclose(
        env.episode_metrics["raw_action_abs_mean"], env.actions.abs().mean(dim=-1)
    )
    assert torch.allclose(
        env.episode_metrics["envelope_limits_active_ratio"],
        (limits > 0).float(),
    )
    assert torch.allclose(
        env.episode_metrics["oracle_unsafe_before_ratio"], (oracle_hard_raw > 0).float()
    )
    assert torch.allclose(
        env.episode_metrics["oracle_unsafe_ratio"], (oracle_hard > 0).float()
    )
    assert torch.allclose(
        env.episode_metrics["oracle_potential"], potential_reward(oracle, low, high)
    )
    oracle_sq = (
        normalized_envelope_params(env.actions_mapped, low, high)
        - normalized_envelope_params(oracle, low, high)
    ) ** 2
    for j, part in enumerate((
        "front_width", "middle_width", "back_width",
        "forward_limit", "backward_limit",
    )):
        assert torch.allclose(
            env.episode_metrics[f"oracle_mse_{part}"], oracle_sq[:, j]
        )


def test_compute_rewards_accumulates_over_steps():
    env = _bare_env()
    env._compute_rewards()
    first = {k: v.clone() for k, v in env.episode_sums.items()}
    env._compute_rewards()
    for k, v in env.episode_sums.items():
        assert torch.allclose(v, 2 * first[k])
