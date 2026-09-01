"""采集真机（cascade sim）行走时的俯仰/横滚/高度轨迹。

产物 /tmp/attitude_traj.npz：50 Hz 时序，供加噪函数设计（谱分析/拟合/回放）。
分段：0-300 站立 | 300-1200 直行 vx=1.0 | 1200-1500 行走+转向 wy=0.5。
开阔场（clear radius 拉满）保证全程平地、无避障扰动 → 纯步态姿态信号。
"""

import isaacgym  # noqa: F401

import sys

import numpy as np
import torch

if not any(arg.startswith("--task") for arg in sys.argv):
    sys.argv += ["--task", "el4090_cascade"]
if "--headless" not in sys.argv:
    sys.argv.append("--headless")

import legged_gym.envs.el_4090.envelope_cascade_83  # noqa: F401,E402
from legged_gym.utils import get_args, task_registry  # noqa: E402

TOTAL = 1500


def main() -> int:
    env_cfg, _ = task_registry.get_cfgs(name="el4090_cascade")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 40.0
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.terrain_proportions = [0.0] * 7 + [0.5, 0.5]
    env_cfg.terrain.pillar_center_clear_radius = 20.0  # 开阔场
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.resampling_time = 99999

    args = get_args()
    env, _ = task_registry.make_env(name="el4090_cascade", args=args, env_cfg=env_cfg)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))

    policy = torch.jit.load(
        str(env.cfg.se2_policy.checkpoint).format(
            LEGGED_GYM_ROOT_DIR="/home/t3chichi/el4090_legged_gym/legged_gym"
        ),
        map_location=env.device,
    )
    policy.eval()

    rows = []
    for step in range(TOTAL):
        if step < 300:
            vx, wy = 0.0, 0.0
        elif step < 1200:
            vx, wy = 1.0, 0.0
        else:
            vx, wy = 1.0, 0.5
        env.commands[:, 0] = vx
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = wy
        if step == 0:
            env.compute_observations()
        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        g = env.projected_gravity[0]
        rows.append([
            step, env.dt * step, vx, wy,
            float(g[0]), float(g[1]), float(g[2]),
            float(env.base_pos[0, 2]),
            float(env.base_lin_vel[0, 0]), float(env.base_lin_vel[0, 1]),
            float(env.base_ang_vel[0, 2]),
            int(dones.sum().item()),
        ])

    arr = np.asarray(rows, dtype=np.float64)
    np.savez("/tmp/attitude_traj.npz",
             t=arr[:, 1], cmd_vx=arr[:, 2], cmd_wy=arr[:, 3],
             gx=arr[:, 4], gy=arr[:, 5], gz=arr[:, 6],
             height=arr[:, 7], vx=arr[:, 8], vy=arr[:, 9], wz=arr[:, 10])
    dones = int(arr[:, 11].sum())
    print(f"记录完成: {TOTAL} 步 ({TOTAL * 0.02:.0f}s), dones={dones}")
    print(f"  gx 范围 [{arr[:, 4].min():+.4f}, {arr[:, 4].max():+.4f}]")
    print(f"  height 范围 [{arr[:, 7].min():.3f}, {arr[:, 7].max():.3f}]")
    return 0 if dones == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
