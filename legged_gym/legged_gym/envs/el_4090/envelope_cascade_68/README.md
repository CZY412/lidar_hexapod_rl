# envelope_cascade_68：EA2 点云感知 → SE2 包络步态 合并任务（冻结 68 维契约）

将两个已训练模块串成一条 50 Hz 推理链：

```
187 通道 range image ─┐
                      ├─ EA2 GRU 策略(190→5 raw action) ─→ 桥(a→params5→cond8)
机体系 ego-motion ────┘                                        │
                                                              ▼
                              set_envelope_condition(8 维, derive_priors=True)
                                                              │
                              HAA 范围网络 + morphology preset + 68 维 SE2 obs
                                                              ▼
                                        SE2 步态策略(68→18) ─→ 机器人
```

本包**自注册** `el4090_cascade_68` 任务（导入即注册，未改动 `envs/__init__.py`
等任何包外文件）。`ea2.enable=False` 可一键回退纯 SE2 行为（随机包络）。

与姊妹任务 `envelope_cascade_83`（冻结旧 83 维契约的遗留合并演示）互为
**结构孪生**：两包完全独立、互不导入，权重不可互换。

## 文件结构

```text
envelope_cascade_68/
├── __init__.py               # 任务注册（幂等）
├── el4090_cascade_config.py  # El4090Cascade68Cfg / El4090Cascade68CfgPPO
├── el4090_cascade_env.py     # EL_4090_CASCADE_68(EL_4090_SE2_68)
├── se2_frozen/               # 冻结 68 维 SE2 层（env/config/envelope_condition，
│                             #   源自 spider_envelop_2@84c08ca 逐字拷贝，勿改动）
├── ea2_perception.py         # 187 射线 reduced raycast + 10Hz 时钟 + obs190
├── ea2_policy.py             # ActorCriticRecurrent 手工构造 + strict 加载
├── envelope_bridge.py        # raw a5 → params5 → condition8
├── checkpoints/              # 权重集中管理（见其 README；policy_1.pt 占位未就位）
└── README.md
scripts/play_cascade_68.py    # 键盘演示（SE2 68 维 checkpoint 就位后可玩）
tests/ea2/cascade_68/         # 契约测试 + 感知测试 + Isaac 环境级测试 + 闭环
```

## 依赖锚点（当前代码状态）

| 依赖 | 锚点 |
|---|---|
| EA2 感知权重 | `checkpoints/ea2_envelope_v3att.pt` ← `logs/el4090_ea2/v3_attitude/model_0.pt`（md5 `d294151c…`，折叠标度 k=0.11875）；回滚备用 `ea2_envelope.pt`（v2_multik，md5 `4716b023…`） |
| SE2 步态策略 | **【占位】** `checkpoints/policy_1.pt`（TorchScript，68→18，MLP[512,256,128]）——待 68 维主线 `el4090_envelop_2` 训练产物；就位后 md5 由契约测试钉死 |
| HAA 范围网络 | `checkpoints/haa_range.pt`（`spider_envelop_2/envelop_network/haa_range.pt` 的逐字节拷贝，md5 `640627fd…`，配置指针已覆盖为包内路径） |
| EA2 感知代码 | `envelope_adaptive_2/` 的 `airy_mount.load_selected_channels` / `range_image.build_selected_range_image` / `el_4090_ea2_env.assemble_observation·map_actions_to_params·refresh_range_image_from_scan` / `envelope_geometry.envelope_params_to_condition` / `LidarSensor.apply_noise` / `LidarWarpKernels.draw_optimized_kernel_pointcloud`（活引用，签名漂移由契约/感知测试捕获） |
| SE2 步态环境代码 | `se2_frozen/` 冻结拷贝（源自 `spider_envelop_2@84c08ca` 逐字拷贝；68 维观测契约由 `tests/ea2/cascade_68/test_cascade68_contracts.py` 钉死，上游 SE2 再演进不影响本任务。v1 基座 `spider_envelop/` 与 `utils/envelop/network/haa_swing_range` 为共享引用） |
| 187 通道表 | `envelope_adaptive_2/selected_airy_channels.pt`（训练/部署共用） |

## 关键设计（与训练语义逐项对齐，同 cascade_83）

- **感知节奏**：range image 每 5 步（10 Hz）刷新、首步即刷、策略 50 Hz 消费；
  reset 的 env 置空帧（=range_max）直到下一次全局扫描——EA2 空帧契约。
- **感知路径**：EA2 训练用的 reduced raycast（`wp.launch` 直发 187 射线）；
  Warp mesh 由 Isaac trimesh 地形构建，顶点 `x/y -= border_size` 对齐 env 坐标。
- **传感器位姿**：默认 full `base_quat`（物理正确）；`ea2.yaw_only=True`
  复现 EA2 训练位姿，用于归因 pitch/roll OOD。
- **ego-motion**：measured 机体系 `[vx, vy, wz] / (1.5, 1.0, 1.5)`。
- **动作映射**：`map_actions_to_params`（仓库唯一真相）+ live EA2 配置的
  soft/action_max；`ea2.fold_scale=0.11875` 断言防陈旧折叠（构造期校验）。
- **条件注入**：`set_envelope_condition(derive_priors=True)`（幂等），自动
  刷新 HAA 范围与 morphology preset；SE2 obs 在同一控制步内重建（obs 的
  `dof_pos` 以该预设为中心，故虽不观测包络本身，obs 仍依赖条件）。
- **隐状态**：EA2 GRU stateful 推理，对本步 done 的 env `policy.reset(dones)`。
- **reset 语义**：`_resample_commands` 只覆盖条件来源——出生默认最大包络
  （`ea2.reset_condition="max"|"midpoint"`）；其余复用 SE2 自带 reset 链。

## 使用

```bash
conda activate el4090; PATH=/home/t3chichi/anaconda3/envs/el4090/bin:$PATH
cd el4090_legged_gym

# 演示（需先放置 68 维 SE2 策略 policy_1.pt，见 checkpoints/README.md）
python legged_gym/legged_gym/scripts/play_cascade_68.py --task=el4090_cascade_68 --num_envs 1

# 测试（不需要 SE2 策略）
legged_gym/tests/ea2/cascade_68/run_cascade68_tests.sh
#   [1/4] pytest：契约 + 感知（平地解析对齐 / 单柱方向性 / 噪声 / 节奏 / 桥幂等）
#   [2/4] env 集成主变体（T0 断言 + 10Hz 节奏 + reset/stale + 全链路有限性）
#   [3/4] env 集成 --ea2-off（纯 SE2 回退，随机包络）
#   [4/4] env 集成 --birth-condition midpoint（出生预设旋钮）
python legged_gym/tests/ea2/cascade_68/run_cascade68_closed_loop_test.py
#   闭环验收（站立稳定 + vx=1.0 行走推进）——需 policy_1.pt 就位
```

Isaac 约束：单进程只构造一个 env 实例（与 EA2/cascade_83 相同）。

## 当前验收状态（2026-09-04）

- 契约 + 感知 pytest：**20 passed**（含 68 维契约钉死、三权重指针包内断言、
  haa/EA2 md5 硬钉；policy_1.pt 占位期 skipif）。
- 全量 ea2 回归：**287 passed, 10 skipped**（= 基线 267 + 本包新增 20，零回归）。
- env 集成三变体（main / --ea2-off / midpoint）：**全部 PASS**（68 维 T0、
  10Hz 节奏 26/26、reset-stale、桥接有限性、出生预设）。
- 闭环：占位行为已验证（policy_1.pt 缺失 → exit 1 + 指引）；**行走验收待
  68 维 SE2 权重就位后执行**。

## 已知局限（记录，不在代码里规避）

1. **policy_1.pt 未就位**：68 维 SE2 步态权重尚未训练；play 与闭环验收在此
   之前不可用（占位行为见 checkpoints/README.md）。
2. **EA2 训练域**：vx∈[0,1]、vy≈0、平地+柱阵。演示可自由给指令，超域属 OOD。
3. **姿态/高度 OOD**：EA2 训练时机体高度恒 0.52、无 pitch/roll；`ea2.yaw_only`
   开关用于归因；根治需在 EA2 训练中启用 SwayCfg/HeightCfg 随机化。
4. **地形几何差**：Isaac 侧柱子是 0.1 m heightfield 台阶面，EA2 训练地图是
   精确盒体 mesh，横向命中存在 ±cm 级偏差。
5. **187 通道视野**：仅前方 x∈[0.65,3.65]、y∈[-1,1]，侧后障碍依赖 GRU 记忆。
6. **出生净空语义（承自 cascade_83）**：`pillar_center_clear_radius=2.5` 是
   中心距语义。注意 cascade_83 的 README 声称"取 4.0"与其代码（恒为 2.5）
   存在文档漂移；本包沿用代码值 2.5 以保持与已验收行为一致，但按其注释的
   几何论证（4m 柱最坏半对角 ≈2.83m），如遇出生点被柱身覆盖可改 4.0
   （柱身边缘距出生点 ≥1.17m）。
7. **可视化滞后一帧**：包络六边形/点云在 `super().post_physics_step()` 内
   绘制，反映的是上一步的条件（演示观感无碍）。
8. **冻结层默认值为主线共享（逐字 vendor 的结果）**：`se2_frozen/config.py`
   继承的 PPO `runner.experiment_name` 与冻结 env 的默认 `task_name`
   （`el_4090_envelop_2_p_haa_range` / `el4090_envelop_2`）与 SE2 主线同值。
   冻结层**不是训练入口**（未注册 task_registry）；如需训练必须走级联任务
   （其 CfgPPO 已隔离实验目录 `el4090_cascade_68_p_haa_range`），禁止直接以
   冻结 config 训练，以免 83/68 维产物混入主线日志。`envelope_cascade_83`
   同构（其隔离说明见其自身文档）。
