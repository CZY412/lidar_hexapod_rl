"""多速度姿态扫频采集：vx ∈ {0.25,0.5,0.75,1.0,1.25,1.5}，每档先 reset、
丢弃 50 步过渡、录 400 步（20s）。开阔场保证全程平地。

产物 /tmp/attitude_speed_sweep.npz（t, speed, gx, gy, gz, height, vx, vy, wz）。
"""

import isaacgym  # noqa: F401

import sys

import numpy as np
import torch

if not any(arg.startswith("--task") for arg in sys.argv):
    sys.argv += ["--task", "el4090_cascade_83"]
if "--headless" not in sys.argv:
    sys.argv.append("--headless")

import legged_gym.envs.el_4090.envelope_cascade_83  # noqa: F401,E402
from legged_gym.utils import get_args, task_registry  # noqa: E402

SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
TRANSITION = 50
RECORD = 400


def main() -> int:
    env_cfg, _ = task_registry.get_cfgs(name="el4090_cascade_83")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 1e9
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.terrain_proportions = [0.0] * 7 + [0.5, 0.5]
    env_cfg.terrain.pillar_center_clear_radius = 20.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.resampling_time = 99999

    args = get_args()
    env, _ = task_registry.make_env(name="el4090_cascade_83", args=args, env_cfg=env_cfg)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))

    policy = torch.jit.load(
        str(env.cfg.se2_policy.checkpoint).format(
            LEGGED_GYM_ROOT_DIR="/home/t3chichi/el4090_legged_gym/legged_gym"
        ),
        map_location=env.device,
    )
    policy.eval()

    rows = []
    total_dones = 0
    step = 0
    for speed in SPEEDS:
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        for k in range(TRANSITION + RECORD):
            env.commands[:, 0] = speed
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            if step == 0:
                env.compute_observations()
            obs = env.get_observations()
            with torch.no_grad():
                actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())
            total_dones += int(dones.sum().item())
            if k >= TRANSITION:
                g = env.projected_gravity[0]
                rows.append([
                    env.dt * step, speed,
                    float(g[0]), float(g[1]), float(g[2]),
                    float(env.base_pos[0, 2]),
                    float(env.base_lin_vel[0, 0]),
                    float(env.base_lin_vel[0, 1]),
                    float(env.base_ang_vel[0, 2]),
                ])
            step += 1

    arr = np.asarray(rows, dtype=np.float64)
    np.savez("/tmp/attitude_speed_sweep.npz",
             t=arr[:, 0], speed=arr[:, 1],
             gx=arr[:, 2], gy=arr[:, 3], gz=arr[:, 4],
             height=arr[:, 5], vx=arr[:, 6], vy=arr[:, 7], wz=arr[:, 8])
    print(f"扫频完成: {len(SPEEDS)} 档 × {RECORD} 步, dones={total_dones}")
    print("\nspeed | pitch(mean±std, °) | roll std(°) | height mean±std(m) | 实测vx")
    for speed in SPEEDS:
        m = arr[:, 1] == speed
        p = np.degrees(np.arcsin(np.clip(arr[m, 2], -1, 1)))
        r = np.degrees(np.arcsin(np.clip(arr[m, 3], -1, 1)))
        print(f"  {speed:.2f} | {p.mean():+5.2f} ± {p.std():.2f} | {r.std():.2f} | "
              f"{arr[m, 5].mean():.3f} ± {arr[m, 5].std():.4f} | {arr[m, 6].mean():.2f}")
    return 0 if total_dones == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
