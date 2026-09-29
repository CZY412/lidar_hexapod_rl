"""Headless integration test for direct-HAA-range ``el4090_envelop_3``.

Run from ``legged_gym``:

    python tests/test_envelop_3_env.py --headless

The test deliberately exercises both direct random sampling and a coupled
extreme range.  It is a simulator smoke test, not a policy-quality benchmark.
"""

import isaacgym  # noqa: F401 -- must precede torch-backed environment imports.
import torch

from legged_gym.envs import *  # noqa: F401,F403 -- register tasks.
from legged_gym.utils import get_args, task_registry


TASK_NAME = "el4090_envelop_3"
PAIR_TO_LEG = torch.tensor((0, 1, 2, 0, 1, 2), dtype=torch.long)


def _assert_valid_ranges(env, ranges: torch.Tensor, label: str) -> None:
    low, high = env.cfg.haa_range.joint_limits
    min_width = env.cfg.haa_range.min_width
    assert ranges.shape == (env.num_envs, 6, 2), f"{label}: unexpected shape {tuple(ranges.shape)}"
    assert bool(torch.all(ranges[..., 0] >= low)), f"{label}: lower endpoint below {low}"
    assert bool(torch.all(ranges[..., 1] <= high)), f"{label}: upper endpoint above {high}"
    assert bool(torch.all(ranges[..., 1] - ranges[..., 0] >= min_width - 1e-6)), (
        f"{label}: interval narrower than {min_width}"
    )
    assert torch.equal(ranges[:, 0], ranges[:, 3]), f"{label}: LB/RB ranges differ"
    assert torch.equal(ranges[:, 1], ranges[:, 4]), f"{label}: LF/RF ranges differ"
    assert torch.equal(ranges[:, 2], ranges[:, 5]), f"{label}: LM/RM ranges differ"


def _extreme_ranges(env) -> torch.Tensor:
    """One valid, coupled mixed-endpoint case shared with play's stress presets."""
    pair_ranges = torch.tensor(
        ((-1.30, -0.95), (0.95, 1.30), (-0.15, 0.15)),
        dtype=torch.float,
        device=env.device,
    )
    return pair_ranges[PAIR_TO_LEG.to(env.device)].unsqueeze(0).repeat(env.num_envs, 1, 1)


def _check_standing_command_sampling(env, rounds: int = 128) -> None:
    """Check that resampling supplies exact-zero and near-zero stand examples."""
    exact_zero_count = 0
    standing_count = 0
    total = 0
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    threshold = float(env.cfg.rewards.stand_command_threshold)
    for _ in range(rounds):
        env._resample_commands(env_ids)
        commands = env.commands[:, :3]
        exact_zero_count += int(torch.all(commands == 0.0, dim=1).sum().item())
        standing_count += int((torch.linalg.vector_norm(commands, dim=1) < threshold).sum().item())
        total += env.num_envs

    # Expected rates are 2.5% exact-zero and 5% total standing. These broad
    # bounds avoid statistical flakes while detecting removed sampling logic.
    exact_fraction = exact_zero_count / total
    standing_fraction = standing_count / total
    assert 0.005 <= exact_fraction <= 0.06, f"exact-zero fraction {exact_fraction:.3f} is unexpected"
    assert 0.02 <= standing_fraction <= 0.10, f"standing fraction {standing_fraction:.3f} is unexpected"


def test_env(args) -> None:
    """Create v3, validate its direct-range contracts, and run zero-action steps."""
    env_cfg, _ = task_registry.get_cfgs(name=TASK_NAME)
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.envelope_debug_viz = False

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()
    assert obs.shape == (env.num_envs, 79), f"unexpected observation shape {tuple(obs.shape)}"
    assert bool(torch.isfinite(obs).all()), "initial observation contains NaN or Inf"

    sampled = env.sample_haa_ranges()
    _assert_valid_ranges(env, sampled, "direct sample")
    centers = 0.5 * (sampled[..., 0] + sampled[..., 1])
    target = env._get_condition_target_dof_pos()[:, env.haa_indices]
    assert torch.allclose(target, centers, atol=1e-6), "HAA centre bias does not match sampled interval centres"

    extreme = _extreme_ranges(env)
    applied = env.set_haa_ranges(extreme)
    _assert_valid_ranges(env, applied, "mixed extreme sample")
    assert torch.equal(applied, extreme), "set_haa_ranges changed a valid extreme range"

    _check_standing_command_sampling(env)
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(int(args.steps)):
        obs, _, reward, _, _ = env.step(actions)
        assert obs.shape == (env.num_envs, 79), f"unexpected step observation shape {tuple(obs.shape)}"
        assert bool(torch.isfinite(obs).all()), "step observation contains NaN or Inf"
        assert bool(torch.isfinite(reward).all()), "reward contains NaN or Inf"

    print(
        f"PASS {TASK_NAME}: envs={env.num_envs}, steps={args.steps}, "
        f"obs={tuple(obs.shape)}, HAA ranges and standing samples validated."
    )


if __name__ == "__main__":
    args = get_args([
        {"name": "--steps", "type": int, "default": 100, "help": "Zero-action simulator steps after validation."},
    ])
    args.task = TASK_NAME
    # ``get_args`` already owns --num_envs. Keep this smoke test small unless
    # the caller explicitly supplies its built-in override.
    if args.num_envs is None:
        args.num_envs = 3
    test_env(args)
