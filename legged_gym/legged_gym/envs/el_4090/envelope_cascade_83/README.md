# envelope_cascade：EA2 点云感知 → SE2 包络步态 合并演示

将两个已训练任务串成一条 50 Hz 推理链：

```
187 通道 range image ─┐
                      ├─ EA2 GRU 策略(190→5 raw action) ─→ 桥(a→params5→cond8)
机体系 ego-motion ────┘                                        │
                                                              ▼
                              set_envelope_condition(8 维, derive_priors=True)
                                                              │
                              HAA 范围网络 + morphology preset + 83 维 SE2 obs
                                                              ▼
                                        SE2 步态策略(83→18) ─→ 机器人
```

本包**自注册** `el4090_cascade_83` 任务（导入即注册，未改动 `envs/__init__.py`
等任何包外文件）。`ea2.enable=False` 可一键回退纯 SE2 行为（随机包络）。

## 文件结构

```text
envelope_cascade_83/
├── __init__.py               # 任务注册（幂等）
├── el4090_cascade_config.py  # El4090Cascade83Cfg / El4090Cascade83CfgPPO
├── el4090_cascade_env.py     # EL_4090_CASCADE_83(EL_4090_SE2_83)
├── ea2_perception.py         # 187 射线 reduced raycast + 10Hz 时钟 + obs190
├── ea2_policy.py             # ActorCriticRecurrent 手工构造 + strict 加载
├── envelope_bridge.py        # raw a5 → params5 → condition8
├── checkpoints/              # EA2 感知权重（pin 进 git，见其 README）
└── README.md
scripts/play_cascade.py       # 键盘演示（SE2 大模型 checkpoint 就位后可玩）
tests/ea2/cascade/            # 契约测试 + Isaac 环境级测试
```

## 依赖锚点（当前代码状态）

| 依赖 | 锚点 |
|---|---|
| EA2 感知权重 | `checkpoints/ea2_envelope.pt` ← `logs/el4090_ea2/v2_multik/model_0.pt`（md5 `4716b023…`，折叠标度 k=0.11875） |
| SE2 步态策略 | `checkpoints/policy_1.pt`（TorchScript，83→18；`torch.jit.load` 直接消费，不依赖 rsl_rl/logs） |
| HAA 范围网络 | `checkpoints/haa_range.pt`（`spider_envelop_2/envelop_network/haa_range.pt` 的逐字节拷贝，配置指针已覆盖为包内路径） |
| EA2 感知代码 | `envelope_adaptive_2/` 的 `airy_mount.load_selected_channels` / `range_image.build_selected_range_image` / `el_4090_ea2_env.assemble_observation·map_actions_to_params·refresh_range_image_from_scan` / `envelope_geometry.envelope_params_to_condition` / `LidarSensor.apply_noise` / `LidarWarpKernels.draw_optimized_kernel_pointcloud` |
| SE2 步态环境代码 | `spider_envelop_2/` 全继承（条件状态/HAA/P 控制未改动） |
| 187 通道表 | `envelope_adaptive_2/selected_airy_channels.pt`（训练/部署共用） |

策略权重采取**集中管理**：三个权重都 pin 在 `checkpoints/`，运行与测试均不依赖
`logs/` 或其他任务目录；同步流程见 `checkpoints/README.md`。以上 EA2/SE2
**代码**均为引用而非复制，上游函数签名变化会被 `tests/ea2/cascade/` 捕获。

## 关键设计（与训练语义逐项对齐）

- **感知节奏**：range image 每 5 步（10 Hz）刷新、首步即刷、策略 50 Hz 消费；
  reset 的 env 置空帧（=range_max）直到下一次全局扫描——EA2 空帧契约。
- **感知路径**：EA2 训练用的 reduced raycast（`wp.launch` 直发 187 射线），
  非 LidarSensor 注入；Warp mesh 由 Isaac trimesh 地形构建，顶点
  `x/y -= border_size` 对齐 env 坐标（env_origins 不含 border）。
- **传感器位姿**：默认 full `base_quat`（物理正确）；`ea2.yaw_only=True`
  复现 EA2 训练位姿，用于归因 pitch/roll OOD。
- **ego-motion**：measured 机体系 `[vx, vy, wz] / (1.5, 1.0, 1.5)`，与
  EA2 观测语义一致。
- **动作映射**：`map_actions_to_params`（仓库唯一真相）+ live EA2 配置的
  soft/action_max；`ea2.fold_scale` 断言防止陈旧折叠（构造期校验）。
- **条件注入**：`set_envelope_condition(derive_priors=True)`（幂等），
  自动刷新 HAA 范围与 morphology preset；SE2 obs 在同一控制步内重建。
- **隐状态**：EA2 GRU stateful 推理，env 在 `post_physics_step` 末尾对本步
  done 的 env `policy.reset(dones)`（与 `play_ea2.py` 部署模式一致）。
- **reset 语义**：`_resample_commands` 只覆盖条件来源——出生默认最大包络
  （`ea2.reset_condition="max"|"midpoint"`），EA2 同一步即接管；关节回位、
  GPU 同步、HAA 刷新全部复用 SE2 自带的 reset 链。

## 使用

```bash
conda activate el4090; PATH=/home/t3chichi/anaconda3/envs/el4090/bin:$PATH
cd el4090_legged_gym

# 演示（策略已全部 pin 在 checkpoints/，无外部依赖）
python legged_gym/legged_gym/scripts/play_cascade.py --task=el4090_cascade_83 --num_envs 1
#   w/s/a/d/q/e 移动转向, 1/2/3 档位, x/空格 急停, v 点云/包络可视化, ESC 退出
#   --max_steps N 限步运行（headless 冒烟用）

# 测试（不需要交互）
legged_gym/tests/ea2/cascade/run_cascade_tests.sh
#   [1/4] pytest：契约 + 感知（平地解析对齐 / 单柱方向性 / 噪声 / 节奏 / 桥幂等）
#   [2/4] env 集成主变体（T0 断言 + 10Hz 节奏 + reset/stale + 全链路有限性）
#   [3/4] env 集成 --ea2-off（纯 SE2 回退，随机包络）
#   [4/4] env 集成 --birth-condition midpoint（出生预设旋钮）
python legged_gym/tests/ea2/cascade/run_cascade_closed_loop_test.py
#   闭环验收：SE2 jit 策略 + cascade 全链路（站立稳定 + vx=1.0 行走推进）
```

Isaac 约束：单进程只构造一个 env 实例（与 EA2 相同的 segfault 约束）。

## 当前验收状态（2026-09-01，点云/六边形修复后）

- 闭环验收 `run_cascade_closed_loop_test.py`：站立 mean|vx|=0.047、vx=1.0 行走
  前 2.5s mean vx=0.932、柱阵中 8s 零终止、EA2 包络全程响应、hexagon 无漂移
  断言通过（顶点机体系范围精确匹配条件）。
- 全套件 `run_cascade_tests.sh` exit=0；完整 ea2 回归套件无回归。
- 可视化修复记录：调试点云曾缺传感器系→机体系转换（漏安装旋转，"点云朝天"
  的根因），已按 EA2 reduced 路径同构修复并有平地落点断言；六边形顶点约定与
  EA2 `compute_hex_vertices` 逐点一致，无合并偏差。
- 已知演示侧注意点：构造期初始姿态与地形穿插（上游既有行为），演示与测试
  在 make_env 后立即做一次正规 `reset_idx` 落地；站立时若 EA2 收窄包络，
  形态预设变化会带来姿态调整，属设计内行为。
- **出生净空语义**：`pillar_center_clear_radius` 是中心距语义（柱子中心距
  tile 中心 ≥ 此值即可放置），不是净空区——必须 ≥ 柱子最坏半对角（4m 柱
  ≈2.83m），当前取 4.0（柱身边缘距出生点 ≥1.17m）。取 2.0 时柱身仍会盖住
  出生点（实证：出生高度 1.91m = 柱顶，机器人被顶上柱顶循环摔落）。

## 已知局限（记录，不在代码里规避）

1. **EA2 训练域**：vx∈[0,1]、vy≈0、平地+柱阵。演示可自由给指令，超域属 OOD
   （T7 指令域扫描仅记录不修复）。
2. **姿态/高度 OOD**：EA2 训练时机体高度恒 0.52、无 pitch/roll；真机步态下
   高度 0.53–0.64 且有姿态摆动。`ea2.yaw_only` 开关用于归因；根治需在 EA2
   训练中启用 `_contracts.py` 预留的 SwayCfg/HeightCfg 随机化。
3. **地形几何差**：Isaac 侧柱子是 0.1 m heightfield 台阶面，EA2 训练地图是
   精确盒体 mesh，横向命中存在 ±cm 级偏差。
4. **187 通道视野**：仅前方 x∈[0.65,3.65]、y∈[-1,1]，侧后障碍依赖 GRU 记忆
   （EA2 README §8 的已知残余，长时贴障静止会缓慢张开）。
5. **可视化滞后一帧**：包络六边形/点云在 `super().post_physics_step()` 内
   绘制，反映的是上一步的条件（演示观感无碍）。
6. **P_LOWPASS / 输出速率限制**：预留开关未实现——`cfg.control.control_type
   ="P_LOWPASS"` 可直接启用继承链自带的低通（tau 走
   `default_dof_pos_filter_tau`）；输出速率限制（RateLimitedOracle 语义）
   如 T6 A/B 显示包络抖动再加。
