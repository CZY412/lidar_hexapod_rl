"""Closed-loop acceptance test: SE2 jit policy + EA2 cascade, full pipeline.

Requires the pinned SE2 policy (``checkpoints/policy_1.pt``).  Runs the real
50 Hz loop — SE2 policy → P control → physics → EA2 perception/policy →
envelope bridge → SE2 observation — and asserts:

* standing: pipeline finite/bounded, robot roughly stationary;
* walking (vx = 1.0 m/s): the robot actually translates forward
  (mean body-frame vx over the first 2.5 s > 0.3 m/s), the EA2 envelope
  reacts over the run, and every output stays finite and in bounds.

One Isaac env per process.  Run:
    python legged_gym/tests/ea2/cascade/run_cascade_closed_loop_test.py
"""

import isaacgym  # noqa: F401  -- must precede all legged_gym imports

import os
import sys

import torch

if not any(arg.startswith("--task") for arg in sys.argv):
    sys.argv += ["--task", "el4090_cascade_83"]
if "--headless" not in sys.argv:
    sys.argv.append("--headless")

import legged_gym.envs.el_4090.envelope_cascade_83  # noqa: F401,E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.envelop.network.haa_swing_range import (  # noqa: E402
    load_envelope_condition_spec,
)
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c  # noqa: E402

STAND_STEPS = 100
WALK_STEPS = 300


def main() -> int:
    failures = []

    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
        if not cond:
            failures.append(name)

    env_cfg, _ = task_registry.get_cfgs(name="el4090_cascade_83")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 20.0  # no timeout inside the test window
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.terrain_proportions = [0.0] * 7 + [0.5, 0.5]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.resampling_time = 99999

    args = get_args()
    env, _ = task_registry.make_env(name="el4090_cascade_83", args=args, env_cfg=env_cfg)
    # 构造期初始姿态与地形穿插（上游既有行为）：先做一次正规 reset。
    env.reset_idx(torch.arange(env.num_envs, device=env.device))

    se2_ckpt = str(env.cfg.se2_policy.checkpoint).format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    check("SE2 policy pinned", os.path.exists(se2_ckpt), se2_ckpt)
    if not os.path.exists(se2_ckpt):
        return 1
    policy = torch.jit.load(se2_ckpt, map_location=env.device)
    policy.eval()

    spec = load_envelope_condition_spec(ea2c.ENVELOPE_SPEC_CONFIG_PATH)
    low5 = torch.tensor(spec.low[:5], device=env.device)
    high5 = torch.tensor(spec.high[:5], device=env.device)

    dones_total = 0
    distinct_conditions = set()

    def run_phase(steps, vx_cmd, label):
        nonlocal dones_total
        vx_history = []
        for step in range(steps):
            env.commands[:, 0] = vx_cmd
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            if step == 0:
                env.compute_observations()
            obs = env.get_observations()
            with torch.no_grad():
                actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions.detach())
            dones_total += int(dones.sum().item())

            p5 = env.cascade_params5
            if not (bool(torch.isfinite(p5).all()) and bool(torch.isfinite(obs).all())):
                check(f"[{label}] finite pipeline at step {step}", False)
                return []
            if bool(((p5 < low5 - 1e-5) | (p5 > high5 + 1e-5)).any()):
                check(f"[{label}] params5 in bounds at step {step}", False)
                return []
            distinct_conditions.add(tuple(env.cascade_condition8[0].detach().cpu().tolist()))
            vx_history.append(float(env.base_lin_vel[0, 0]))
        return vx_history

    stand_vx = run_phase(STAND_STEPS, 0.0, "站立")
    if stand_vx:
        mean_stand = sum(abs(v) for v in stand_vx) / len(stand_vx)
        check("站立阶段基本静止", mean_stand < 0.3, f"mean|vx|={mean_stand:.3f}")

    walk_vx = run_phase(WALK_STEPS, 1.0, "行走")
    if walk_vx:
        lead = walk_vx[: min(125, len(walk_vx))]
        mean_lead = sum(lead) / len(lead)
        check("行走阶段真实推进 (前2.5s mean vx > 0.3)", mean_lead > 0.3,
              f"mean vx={mean_lead:.3f}, max={max(walk_vx):.3f}")

    check("EA2 包络在闭环中响应", len(distinct_conditions) >= 3,
          f"{len(distinct_conditions)} distinct")

    # hexagon 漂移断言：SE2 绘制的包络六边形，顶点逆 yaw 转回机体系后必须
    # 落在条件定义的范围框内（x∈[bwd,fwd], |y|≤max width）——排除"画偏"。
    cond = env._get_structure_condition()[0]
    names = list(env.condition_names)
    fwd = float(cond[names.index("forward_limit")])
    bwd = float(cond[names.index("backward_limit")])
    max_w = max(float(cond[names.index(n)]) for n in
                ("front_width", "middle_width", "back_width"))
    hex_world = torch.tensor(
        env._envelope_debug_points_world(0), dtype=torch.float32, device=env.device
    )
    rel = hex_world[:, :2] - env.base_pos[0, :2]
    q = env.base_quat[0]
    yaw = torch.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                      1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    local_x = c * rel[:, 0] - s * rel[:, 1]
    local_y = s * rel[:, 0] + c * rel[:, 1]
    check(
        "hexagon 无漂移（机体系范围匹配条件）",
        bool((local_x >= bwd - 0.03).all() and (local_x <= fwd + 0.03).all()
             and (local_y.abs() <= max_w + 0.03).all()),
        f"x[{float(local_x.min()):.2f},{float(local_x.max()):.2f}]∈[{bwd:.2f},{fwd:.2f}] "
        f"|y|max={float(local_y.abs().max()):.2f}≤{max_w:.2f}",
    )

    print(f"信息: dones 总数 = {dones_total}（柱阵中碰撞属预期行为，仅记录）")

    print("\n===== cascade closed-loop test:",
          "PASS" if not failures else f"FAIL ({failures})", "=====")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
