#!/usr/bin/env python
"""Evaluate whether a trained EA2 policy's envelope varies with the scene.

Loads a checkpoint, steps a batch of environments, and compares the policy's
mapped 5-parameter envelope against the Active-set direct oracle on the same
observations.

Usage (from legged_gym/legged_gym):
    python tests/ea2/eval_ea2_oracle_adaptation.py \
        --task=el4090_ea2 --headless --num_envs=64 \
        --load_run <run_dir> --checkpoint <iter>

Metrics printed:
  * per-parameter policy/std/oracle/std/correlation
  * normalized MAE and overall normalized MSE
  * policy potential vs oracle potential
"""

from __future__ import annotations

import isaacgym  # noqa: F401

import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    normalized_envelope_params,
    potential_reward,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
    compute_direct_oracle_params_with_stats,
)
from legged_gym.utils import get_args, task_registry

_STEPS = 200


def main() -> None:
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    env_cfg.env.num_envs = env_cfg.env.num_envs or 64

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    print(
        f"[eval] policy={type(ppo_runner.alg.policy).__name__} "
        f"recurrent={ppo_runner.alg.policy.is_recurrent}"
    )

    obs = env.get_observations()
    low = env._envelope_low_dev
    high = env._envelope_high_dev

    policy_list = []
    oracle_list = []

    for step in range(_STEPS):
        actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        if ppo_runner.alg.policy.is_recurrent:
            ppo_runner.alg.policy.reset(dones)

        oracle_params, _ = compute_direct_oracle_params_with_stats(
            env.heading,
            env.base_pos[:, :2],
            env.distance_field,
            low,
            high,
            margin=float(env.cfg.envelope.oracle_margin),
            step=float(env.cfg.envelope.oracle_step),
            max_dist=float(env.cfg.envelope.oracle_max_dist),
            soft_dof_pos_limit=float(env.cfg.envelope.soft_dof_pos_limit),
            # must match the training reward's oracle, otherwise the
            # policy-vs-oracle metrics are biased by the interp mismatch
            interp_crossing=bool(
                getattr(env.cfg.envelope, "oracle_interp_crossing", True)
            ),
        )

        policy_list.append(env.actions_mapped.detach().clone())
        oracle_list.append(oracle_params.detach().clone())

    policy_params = torch.cat(policy_list, dim=0).float().cpu().numpy()  # (N,5)
    oracle_params = torch.cat(oracle_list, dim=0).float().cpu().numpy()  # (N,5)

    names = [
        "front_width",
        "middle_width",
        "back_width",
        "forward_limit",
        "backward_limit",
    ]

    print("\n=== per-parameter adaptation metrics ===")
    for i, name in enumerate(names):
        p = policy_params[:, i]
        o = oracle_params[:, i]
        corr = np.corrcoef(p, o)[0, 1] if np.std(p) > 1e-6 and np.std(o) > 1e-6 else float("nan")
        print(
            f"{name:15s} policy_std={np.std(p):.4f} oracle_std={np.std(o):.4f} "
            f"corr={corr:+.3f} policy_min={p.min():.4f} policy_max={p.max():.4f}"
        )

    # Normalized error metrics (same normalization used in reward)
    policy_t = torch.from_numpy(policy_params).float()
    oracle_t = torch.from_numpy(oracle_params).float()
    low_cpu = low.detach().to("cpu")
    high_cpu = high.detach().to("cpu")
    norm_policy = normalized_envelope_params(policy_t, low_cpu, high_cpu)
    norm_oracle = normalized_envelope_params(oracle_t, low_cpu, high_cpu)
    abs_err = (norm_policy - norm_oracle).abs()
    mse = ((norm_policy - norm_oracle) ** 2).mean(dim=-1)

    print("\n=== error metrics ===")
    print(f"normalized MAE per param: {abs_err.mean(dim=0).numpy().round(4)}")
    print(f"overall normalized MSE: {mse.mean().item():.4f}")
    print(f"policy potential mean: {potential_reward(policy_t, low_cpu, high_cpu).mean().item():.4f}")
    print(f"oracle potential mean: {potential_reward(oracle_t, low_cpu, high_cpu).mean().item():.4f}")


if __name__ == "__main__":
    main()
