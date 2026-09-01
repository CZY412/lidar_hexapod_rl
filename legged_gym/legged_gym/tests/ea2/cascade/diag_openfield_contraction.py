"""开阔地假收缩判别实验（因子化对照，只读诊断，不改任何任务/训练内容）。

场地构造性排除记忆尾巴：测试配置把 pillar_center_clear_radius 拉满（=20m），
16m tile 内任何柱子都放不下 → 纯平地开阔场，且机器人 8m 行程内保证无障碍。

变体（每个变体独立进程，单 Isaac env）：
  baseline      真实 raycast + 真实 ego          —— 预期复现假收缩
  yaw_only      无俯仰/横滚的 raycast + 真实 ego  —— 体态经射线几何的因果
  table_range   强制喂训练态开阔地解析签名 + 真实 ego —— 射线几何因果确认
  smooth_ego    真实 raycast + 恒定指令 ego       —— ego 抖动因果
  table_smooth  解析签名 + 恒定 ego              —— 全训练态输入对照
  noise_on      真实输入 + 训练传感器噪声         —— 噪声开关影响

指标：open_score = 5 参数归一化均值（开阔地 oracle 目标 ≈ 1）；
收缩占比/最长收缩段；Δopen_score 与 gx/gy/高度偏差的 Pearson 相关；
range image 分行均值（对照训练开阔地签名）。

运行：python legged_gym/tests/ea2/cascade/diag_openfield_contraction.py --variant <v>
"""

import isaacgym  # noqa: F401

import json
import sys

import torch

if not any(arg.startswith("--task") for arg in sys.argv):
    sys.argv += ["--task", "el4090_cascade_83"]
if "--headless" not in sys.argv:
    sys.argv.append("--headless")

import legged_gym.envs.el_4090.envelope_cascade_83  # noqa: F401,E402
from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (  # noqa: E402
    load_selected_channels,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (  # noqa: E402
    normalized_envelope_params,
)
from legged_gym.utils import get_args, task_registry  # noqa: E402

STAND_STEPS = 60
WALK_STEPS = 340
TOTAL = STAND_STEPS + WALK_STEPS


def main() -> int:
    extra = [
        {"name": "--variant", "type": str, "default": "baseline",
         "help": "baseline|yaw_only|table_range|smooth_ego|table_smooth|noise_on"},
        {"name": "--pillared", "action": "store_true", "default": False,
         "help": "用当前配置的柱阵地形（默认开阔场：clear radius 拉满）"},
    ]
    args = get_args(extra)
    variant = args.variant

    env_cfg, _ = task_registry.get_cfgs(name="el4090_cascade_83")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 20.0
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.terrain_proportions = [0.0] * 7 + [0.5, 0.5]
    # 开阔场：clear radius 拉满 → 16m tile 内柱子一个都放不下（构造性无障碍）；
    # --pillared 时保留配置值（当前 4.0），测真实障碍响应
    if not getattr(args, "pillared", False):
        env_cfg.terrain.pillar_center_clear_radius = 20.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.resampling_time = 99999

    env, _ = task_registry.make_env(name="el4090_cascade_83", args=args, env_cfg=env_cfg)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))

    policy = torch.jit.load(
        str(env.cfg.se2_policy.checkpoint).format(LEGGED_GYM_ROOT_DIR="/home/t3chichi/el4090_legged_gym/legged_gym"),
        map_location=env.device,
    )
    policy.eval()

    perception = env._cascade_perception
    orig_observe = perception.observe

    if variant == "yaw_only":
        env.cfg.ea2.yaw_only = True
    elif variant in ("table_range", "table_smooth"):
        table = load_selected_channels()["slant_ranges"].to(env.device)

        def observe_table(ego):
            perception.range_image[:] = table
            perception.stale[:] = False
            return orig_observe(ego)

        perception.observe = observe_table
    elif variant == "noise_on":
        perception._noise_ctx.sensor_cfg.enable_sensor_noise = True

    if variant in ("smooth_ego", "table_smooth"):
        prev_observe = perception.observe  # 组合变体时须包在 table 补丁之外

        def observe_smooth(ego):
            ego = ego.clone()
            ego[:, 0] = env.commands[:, 0]
            ego[:, 1] = 0.0
            ego[:, 2] = 0.0
            return prev_observe(ego)

        perception.observe = observe_smooth

    bridge = env._cascade_bridge
    low5, high5 = bridge.low5, bridge.high5

    rec = {k: [] for k in ("open", "p5", "rows", "ego", "grav", "height", "dones")}
    dones_total = 0
    for step in range(TOTAL):
        env.commands[:, 0] = 0.0 if step < STAND_STEPS else 1.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        if step == 0:
            env.compute_observations()
        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())
        dones_total += int(dones.sum().item())

        p5n = normalized_envelope_params(env.cascade_params5, low5, high5)[0]
        rec["open"].append(float(p5n.mean()))
        rec["p5"].append(p5n.detach().cpu().tolist())
        ri = perception.range_image[0].view(11, 17)
        rec["rows"].append(ri.mean(dim=1).detach().cpu().tolist())
        rec["ego"].append(env.base_lin_vel[0, :3].detach().cpu().tolist())
        rec["grav"].append(env.projected_gravity[0].detach().cpu().tolist())
        rec["height"].append(float(env.base_pos[0, 2]))
        rec["dones"].append(int(dones.sum().item()))

    # ── 指标 ────────────────────────────────────────────────────────────
    open_s = torch.tensor(rec["open"])
    walk = slice(STAND_STEPS, TOTAL)
    contraction = open_s < 0.9
    longest = cur = 0
    for flag in contraction.tolist():
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    rows_t = torch.tensor(rec["rows"])          # (T, 11)
    grav_t = torch.tensor(rec["grav"])          # (T, 3)
    h_t = torch.tensor(rec["height"])
    d_open = open_s[1:] - open_s[:-1]           # <0 = 收缩方向
    g_t = grav_t[1:, 0]
    r_t = grav_t[1:, 1]
    h_dev = h_t[1:] - h_t[1:].mean()

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        denom = a.std().clamp_min(1e-9) * b.std().clamp_min(1e-9)
        return float((a * b).mean() / denom)

    pillared = bool(getattr(args, "pillared", False))
    tag = variant + ("_pillared" if pillared else "")
    ego_t = torch.tensor(rec["ego"])
    summary = {
        "variant": tag,
        "dones": dones_total,
        "vx_mean_walk": float(ego_t[walk, 0].mean()),
        "open_mean": float(open_s.mean()),
        "open_min": float(open_s.min()),
        "open_mean_walk": float(open_s[walk].mean()),
        "contraction_ratio": float(contraction.float().mean()),
        "longest_contraction_steps": longest,
        "stand_open_mean": float(open_s[:STAND_STEPS].mean()),
        "corr_shrink_gx": corr(-d_open, g_t),
        "corr_shrink_gy": corr(-d_open, r_t),
        "corr_shrink_height": corr(-d_open, h_dev),
        "row_means_last60": [float(v) for v in rows_t[-60:].mean(dim=0)],
        "grav_std_walk": [float(v) for v in grav_t[walk].std(dim=0)],
        "height_std_walk": float(h_t[walk].std()),
    }
    print("===== variant:", tag, "=====")
    for key, value in summary.items():
        if key == "row_means_last60":
            print(f"  {key}: [" + " ".join(f"{v:.2f}" for v in value) + "]")
        elif isinstance(value, list):
            print(f"  {key}: [" + " ".join(f"{v:.3f}" for v in value) + "]")
        else:
            print(f"  {key}: {value}")

    with open(f"/tmp/diag_openfield_{tag}.json", "w") as fh:
        json.dump({"summary": summary, "open": rec["open"], "rows": rec["rows"],
                   "grav": rec["grav"], "height": rec["height"], "p5": rec["p5"]}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
