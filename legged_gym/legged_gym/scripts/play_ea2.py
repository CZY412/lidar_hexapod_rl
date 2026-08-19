#!/usr/bin/env python
"""Play a trained ``el4090_ea2`` recurrent policy with point-cloud viewer.

Usage (from legged_gym/legged_gym):
    python scripts/play_ea2.py --task=el4090_ea2 \
        --load_run <run_dir> --checkpoint <iter> \
        [--sim_device cuda:0] [--headless]

Notes:
- ``--load_run -1`` / ``--checkpoint -1`` load the latest run/checkpoint.
- One environment is used for clarity and LiDAR noise is disabled during play
  (README 2.2.8).
- The GRU hidden state is reset with ``policy.reset(dones)`` after every env
  step, mirroring ``PPO.process_env_step``.
- Red spheres: rays in the Airy mapping table whose noisy distance enters the
  450-dim aggregation.  Green spheres: other real hits.  Cyan hexagon: current
  policy envelope footprint.
"""

import time

import isaacgym  # noqa: F401  (must be imported before torch)
from isaacgym import gymapi
from legged_gym.envs import *  # noqa: F401,F403  (register tasks)
from legged_gym.utils import get_args, task_registry


def _set_camera(env, position, lookat) -> None:
    if env.viewer is None:
        return
    cam_pos = gymapi.Vec3(float(position[0]), float(position[1]), float(position[2]))
    cam_target = gymapi.Vec3(float(lookat[0]), float(lookat[1]), float(lookat[2]))
    env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)


def play(args) -> None:
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Play overrides (README 2.2.8/2.7): one env, sensor noise off, viz on env 0.
    env_cfg.env.num_envs = 1
    env_cfg.lidar.enable_sensor_noise = False
    env_cfg.lidar.debug_env_ids = [0]
    env_cfg.lidar.debug_point_stride = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _set_camera(env, [0.0, -45.0, 30.0], [0.0, 0.0, 0.6])

    # Load the latest (or requested) checkpoint.
    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    print(f"[play_ea2] policy={type(ppo_runner.alg.policy).__name__} "
          f"recurrent={ppo_runner.alg.policy.is_recurrent}")

    obs = env.get_observations()
    step = 0
    print("[play_ea2] running. Ctrl+C to stop.")
    try:
        while True:
            start = time.time()
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            # Recurrent inference: zero GRU hidden for envs that just reset,
            # exactly like PPO.process_env_step does during training.
            if ppo_runner.alg.policy.is_recurrent:
                ppo_runner.alg.policy.reset(dones)

            step += 1
            if step % 10 == 0:
                mapped = env.actions_mapped[0].detach().cpu().numpy()
                print(
                    f"[play_ea2] step={step} rew={float(rews[0]):+.3f} "
                    f"range_min={float(env.range_image[0].min()):.2f} "
                    f"envelope=[{mapped[0]:.3f},{mapped[1]:.3f},{mapped[2]:.3f},"
                    f"{mapped[3]:.3f},{mapped[4]:.3f}]",
                    flush=True,
                )

            # Loose real-time pacing (env.dt = 0.02 s).
            elapsed = time.time() - start
            if elapsed < env.dt:
                time.sleep(env.dt - elapsed)
    except KeyboardInterrupt:
        print(f"\n[play_ea2] stopped after {step} steps.")


if __name__ == "__main__":
    args = get_args()
    play(args)
