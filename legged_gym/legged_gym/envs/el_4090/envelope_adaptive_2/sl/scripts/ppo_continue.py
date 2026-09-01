#!/usr/bin/env python
"""Fine-tune PPO from a supervised-learned policy, with controls (B5).

Three arms are compared:

1. ``scratch``       PPO from a random init (``init_noise_std`` from config)
2. ``sl_init``       PPO initialised from the SL checkpoint, same map seed
3. ``sl_init_cross`` PPO initialised from the SL checkpoint, evaluated on a
                     *different* map seed than the SL model trained on

Arm 3 exists because ``task_registry`` calls ``set_seed(env_cfg.seed)``, which
resets every RNG.  With identical seeds the PPO run replays the same paths the
SL model was trained on, so part of any observed advantage could be path
memorisation rather than weight quality.  Arm 3 separates the two effects.

Implementation notes (learned the hard way)
-------------------------------------------
* ``compute_returns`` must be called **inside** the ``torch.inference_mode()``
  block, exactly as rsl_rl's runner does; otherwise the backward pass fails with
  "Inference tensors cannot be saved for backward".
* Run each arm in its **own process**.  Reusing an env across arms leaves its
  buffers as inference tensors and ``reset()`` then raises
  "Inplace update to inference tensor outside InferenceMode".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.sl.dataset import build_env
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.export import export
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import load_checkpoint
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import SLConfig


def _make_policy(cfg: SLConfig, arm: str, ckpt: Optional[str], num_obs: int, num_actions: int, device: str):
    from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import El4090EA2CfgPPO
    from rsl_rl.modules import ActorCriticRecurrent

    ppo_cfg = El4090EA2CfgPPO()
    if arm == "scratch":
        return ActorCriticRecurrent(
            num_actor_obs=num_obs,
            num_critic_obs=num_obs,
            num_actions=num_actions,
            actor_hidden_dims=list(ppo_cfg.policy.actor_hidden_dims),
            critic_hidden_dims=list(ppo_cfg.policy.critic_hidden_dims),
            activation=ppo_cfg.policy.activation,
            rnn_type=ppo_cfg.policy.rnn_type,
            rnn_hidden_dim=ppo_cfg.policy.rnn_hidden_dim,
            rnn_num_layers=ppo_cfg.policy.rnn_num_layers,
            init_noise_std=ppo_cfg.policy.init_noise_std,
        ).to(device)

    net, meta = load_checkpoint(ckpt, device=device)
    return export(net, cfg, device=device)


def run_arm(
    arm: str,
    seed: int,
    num_envs: int,
    iterations: int,
    ckpt: Optional[str] = None,
    steps_per_env: Optional[int] = None,
    device: str = "cuda",
    log_points=(1, 5, 10, 20, 40, 60, 80, 100),
) -> dict:
    from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import El4090EA2Cfg, El4090EA2CfgPPO
    from rsl_rl.algorithms import PPO

    cfg = SLConfig()
    env_cfg, ppo_cfg = El4090EA2Cfg(), El4090EA2CfgPPO()
    env = build_env(seed, num_envs)
    try:
        num_obs = int(env_cfg.env.num_observations)
        num_actions = int(env_cfg.env.num_actions)
        nsp = int(steps_per_env or ppo_cfg.runner.num_steps_per_env)

        policy = _make_policy(cfg, arm, ckpt, num_obs, num_actions, device)
        algo = PPO(
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
            device=device,
        )
        algo.init_storage("rl", int(num_envs), nsp, [num_obs], [num_obs], [num_actions])

        env.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )
        reset_out = env.reset()
        obs = (reset_out[0] if isinstance(reset_out, tuple) else reset_out).to(device)

        curve = []
        t0 = time.time()
        for it in range(iterations):
            rewards = []
            with torch.inference_mode():
                for _ in range(nsp):
                    actions = algo.act(obs, obs)
                    out = env.step(actions.to(device))
                    obs_new, priv, reward, done, info = out[0], out[1], out[2], out[3], out[4]
                    if priv is None:
                        priv = obs_new
                    obs = obs_new
                    algo.process_env_step(reward, done, info)
                    rewards.append(float(reward.mean()))
                # must stay inside inference_mode, matching rsl_rl's runner
                algo.compute_returns(obs)
            algo.update()

            n = it + 1
            if n in log_points or n == iterations:
                curve.append({"iter": n, "step_reward": float(np.mean(rewards)), "t": time.time() - t0})

        return {
            "arm": arm,
            "seed": seed,
            "steps_per_env": nsp,
            "iterations": iterations,
            "num_envs": num_envs,
            "curve": curve,
            "final_step_reward": curve[-1]["step_reward"] if curve else None,
            "first_step_reward": curve[0]["step_reward"] if curve else None,
        }
    finally:
        del env
        torch.cuda.empty_cache()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["scratch", "sl_init", "sl_init_cross"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--steps-per-env", type=int, default=None,
                    help="override runner.num_steps_per_env (BPTT horizon)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    if args.arm in ("sl_init", "sl_init_cross") and not args.ckpt:
        raise SystemExit("--ckpt is required for sl_init arms")

    res = run_arm(
        arm=args.arm,
        seed=args.seed,
        num_envs=args.num_envs,
        iterations=args.iterations,
        ckpt=args.ckpt,
        steps_per_env=args.steps_per_env,
        device=args.device,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    for pt in res["curve"]:
        print(f"[ppo] {args.arm:<14} iter{pt['iter']:>4}  step_rew={pt['step_reward']:+.4f}  ({pt['t']:.0f}s)")
    print(f"[ppo] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
