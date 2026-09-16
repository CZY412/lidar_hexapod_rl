# EL4090 六足机器人可变自适应包络

[English](README.md) | 简体中文

<div align="center">
  <img src="doc/figures/teaser_cascade83.png" alt="级联策略携自适应包络穿越复杂障碍场" width="49%"/>
  <img src="doc/figures/ea2_gru_pointcloud.png" alt="EA2 从激光点云预测包络" width="49%"/>
</div>

> [!WARNING]
> 本仓库仍在积极开发中。各任务的详细文档放在各自任务包内，可能滞后于代码。
> 真机部署正在推进中，本文展示的结果均为仿真结果。

本仓库面向 EL4090 六足机器人研究**可变自适应包络行为**，构建于扩展版
[legged_gym](https://github.com/leggedrobotics/legged_gym) 之上（见
[仓库基础](#仓库基础)）。机器人的足端可达范围由一个五参数平面**包络**描述：
感知模块根据前方 LiDAR 感知到的障碍对包络进行收放，运动策略在保持足端位于包络
内的同时行走。两个模块分开训练，再串成一条 50 Hz 推理链（**级联 / cascade**）。

<div align="center">
  <video controls muted loop playsinline width="70%">
    <source src="doc/figures/cascade83_demo.mp4" type="video/mp4">
  </video>
  <p><sub><code>envelope_cascade_83</code>：感知、包络与步态全链路穿越复杂障碍场（仿真）。</sub></p>
</div>

## 包络概念

包络是机体系（base yaw）下的一个六边形，由五个长度参数描述（左右对称）：

| 参数 | 范围 (m) | 含义 |
| --- | ---: | --- |
| `front_width` | [0.3, 0.6] | 前缘处允许足端区域的半宽 |
| `middle_width` | [0.3, 0.7] | 中部边缘的半宽 |
| `back_width` | [0.3, 0.6] | 后缘处的半宽 |
| `forward_limit` | [0.6, 0.9] | 足端最前方 x 边界 |
| `backward_limit` | [-0.9, -0.6] | 足端最后方 x 边界（负值） |

五个几何参数加上由它们推导出的三个**形态先验**（每对腿一个，0 = 偏 spider
形态，1 = 偏 mammal 形态）构成所有模块共享的 8 维**包络条件**。形态先验不只
用于惩罚出界落足：它把默认关节姿态在 spider 与 mammal 预设之间插值、把目标
机身高度在 0.53 m 与 0.64 m 之间插值，因此包络直接决定机器人采取的身体形态。

<div align="center">
  <img src="doc/figures/envelope_geometry.png" alt="五参数包络六边形" width="62%"/>
</div>

## 系统架构

系统刻意拆成两个独立训练的模块加一层薄集成：

- **包络模块（感知：点云 → 包络）。** 187 通道 range image（11 × 17 网格，覆盖
  机器人前方 x ∈ [0.65, 3.65] m、y ∈ [-1, 1] m）加 3 维 ego-motion，由 GRU 策略
  输出五个包络参数。感知 10 Hz 刷新，控制 50 Hz 运行；离开传感器视野的障碍
  依靠 GRU 记忆保持。
- **机器人侧模块（包络 → 运动）。** 包络条件设定形态预设（默认关节姿态与机身
  高度），并经一个小型 HAA 范围网络给出各腿外摆关节范围供塑形奖励使用。运动
  策略将自己的观测映射为相对预设的 18 维关节位置残差；包络本身**不**进入其
  观测——因此得到的是鲁棒策略，而不是显式的包络条件化策略。
- **级联（集成）。** 两个模块冻结后串成一条链：range image + ego-motion →
  EA2 GRU 策略（190 → 5）→ 桥接为 8 维条件 → HAA 网络 + 形态预设 → SE2 步态
  策略（68 或 83 → 18）→ 机器人。单个开关（`ea2.enable=False`）即可回退为
  随机包络下的纯 SE2 行为。

## 任务总览

所有任务位于 `legged_gym/legged_gym/envs/el_4090/`。下表列出各任务在
`task_registry` 中注册的名称与策略观测维度。

| 谱系 | 任务 | 注册名 | 策略观测 | 状态 |
| --- | --- | --- | ---: | --- |
| 包络模块 v1 | `envelope_adaptive` | `el4090_ea` | 74 | 融合式解析管线（LiDAR → 避障规划器 → 包络），无可训练包络模型；已被 v2 取代，仅作参考 |
| 包络模块 v2 | `envelope_adaptive_2` | `el4090_ea2` | 190 | **主线。** 可独立训练的感知模块；监督学习管线 + 可选 PPO 微调；权重 `v2_multik`、`v3_attitude` |
| 机器人侧 v1 | `spider_envelop` | `el4090_envelop` | 74 | 旧版 v1：条件可见策略（8 维条件进入观测） |
| 机器人侧 v2 | `spider_envelop_2` | `el4090_envelop_2` | 68 | **主线。** 条件隐藏策略 + HAA 范围网络；68 维步态策略训练推进中（cascade_68 步态权重的来源） |
| 级联 83 维 | `envelope_cascade_83` | `el4090_cascade_83` | 83 | **已完成，当前效果最好。** 冻结 83 维契约，权重全部 pin 定，闭环验收通过 |
| 级联 68 维 | `envelope_cascade_68` | `el4090_cascade_68` | 68 | 更新的 68 维契约；契约 + 感知测试通过；行走验收待 68 维步态权重就位 |

## 包络模块

**v1（`envelope_adaptive`）——融合式，不可独立训练。** 完整 Airy LiDAR 在机器人
环境内部解析处理：角距曲线经样条峰值检测选出避障方向，扇区检测器驱动五参数的
对称分步收缩/扩张律。包络逻辑与运动回路紧耦合，本身没有可训练模型、无法从
数据中改进。仍以 `el4090_ea` 注册，仅作参考保留。

**v2（`envelope_adaptive_2`，"EA2"）——独立可训练的感知模块。** 包络问题被提升
到一个无机器人实体的简化环境中：运动纯脚本化（A* 路径 + 速度调度），状态转移
与动作无关，学习问题因此退化为对几何 oracle 的干净监督回归。

- **Oracle**：由全局距离场直接计算（轴向 march，margin 0.20 m → 扩展侧软上限 →
  active-set 联合收缩 → 快收慢放的限速平滑），在每个几何可行帧上都给出稠密、
  无碰撞的监督目标。
- **学习（主线）**：滑窗 MSE + 多档回忆辅助头（强迫 GRU 记住离开视野的障碍）
  + 距离场上的可微安全损失。已发布配方（`v2_multik`）val R² ≈ 0.75；
  `v3_attitude` 变体在连续体态回放语料上重训（val R² = 0.7375），修复了真机
  俯仰运动导致的假收缩。
- **导出**：把归一化网络输出折叠进 raw action 空间（重写 actor 末层），部署端
  无需任何前/后处理。可选 PPO 微调从导出权重续训。

<div align="center">
  <video controls muted loop playsinline width="70%">
    <source src="doc/figures/ea2_envelope_gru_demo.mp4" type="video/mp4">
  </video>
  <p><sub>EA2 在简化柱阵环境中的表现：红色为 LiDAR 回波，青色包络随前方障碍收放（本任务不含机器人实体）。</sub></p>
</div>

训练配方、评估指标与已知问题的完整说明：
[envelope_adaptive_2/README.md](legged_gym/legged_gym/envs/el_4090/envelope_adaptive_2/README.md)。

## 机器人侧模块

**v1（`spider_envelop`）——条件可见策略。** 8 维包络条件直接进入 74 维观测，
`envelope_constraint` 奖励惩罚足端越出六边形，策略是显式的包络条件化策略。
调试视角可在机器人脚下绘制当前包络的地面轮廓线。

**v2（`spider_envelop_2`）——条件隐藏策略（主线）。** 策略观测收缩为 68 维
（机体速度、重力、速度指令、相对*随条件变化*的默认姿态的关节位置、关节速度、
上一帧动作、步态相位）。包络、形态先验与 HAA 范围都只在环境内部：

- `embedded_state_default_dof_pos` 按形态先验插值默认关节姿态与机身高度，
  因此即使策略看不到包络，控制中心仍随包络移动。
- HAA 范围网络（MLP，8 维输入 → 每腿 `[下限, 上限]`）在包络变化时刷新；其输出
  驱动 `haa_range_violation` 与 `haa_phase_tracking` 塑形奖励。

接口细节、68 维观测的精确布局与 HAA 网络契约：
[envs/el_4090/ReadMe.md](legged_gym/legged_gym/envs/el_4090/ReadMe.md)。

## 级联任务

级联把两个模块冻结后串成一条 50 Hz 链（`ea2_perception` → `envelope_bridge` →
`set_envelope_condition` → HAA 网络 + 形态预设 → 冻结 SE2 策略）。每个级联任务
vendor 了一份冻结的 SE2 环境层（`se2_frozen/`），上游改动无法悄悄破坏已部署的
契约；三个权重（EA2 感知、SE2 步态策略 TorchScript、HAA 网络）都 pin 在
`checkpoints/README.md` 并附 md5 校验。

- **`envelope_cascade_83`**（冻结 83 维契约——83 维观测在 68 维布局上增加形态
  先验与 HAA 范围中心/半宽）是最完整的变体。闭环验收（2026-09-01）通过：站立
  mean |vx| = 0.047，指令 vx = 1.0 时前 2.5 s 平均 vx = 0.932，柱阵中 8 s 零终止。
  键盘演示：`w/s/a/d/q/e` 移动转向，`1/2/3` 步态档位，`x`/空格急停，`v` 切换
  点云/包络可视化，`ESC` 退出。

<div align="center">
  <img src="doc/figures/cascade83_pillar_pass.png" alt="cascade_83 在狭窄缝隙间穿行" width="62%"/>
  <p><sub><code>envelope_cascade_83</code>：步态策略在狭窄缝隙间穿行，EA2 感知保持包络避开前方回波。</sub></p>
</div>

- **`envelope_cascade_68`** 是当前 68 维契约上的结构孪生（完全独立的包；权重与
  83 维任务不可互换）。契约与感知测试（20 项）及三个 env 集成变体全部通过；
  68 维 SE2 步态权重仍在训练中，交互演示与闭环行走验收待其就位。

任务级文档（依赖锚点与验收记录）：
[envelope_cascade_83/README.md](legged_gym/legged_gym/envs/el_4090/envelope_cascade_83/README.md)、
[envelope_cascade_68/README.md](legged_gym/legged_gym/envs/el_4090/envelope_cascade_68/README.md)。

## 工作流与指令

以下命令均从仓库根目录运行。先准备环境：

```bash
conda activate el4090
# Isaac Gym 的 gymtorch JIT 构建需要 ninja 在 PATH 上
export PATH=/home/t3chichi/anaconda3/envs/el4090/bin:$PATH
```

### 包络模块（EA2）——完整管线

管线按 `collect → train → eval → export` 推进，外加可选的第五阶段 PPO 微调；
可用一键管线把全部阶段串联起来。**Isaac Gym 约束：单进程只能建一个环境实例**
——每个地图 seed 单独进程采集。

```bash
EA2_PKG=legged_gym.envs.el_4090.envelope_adaptive_2.sl.scripts
EA2_RUN=legged_gym/legged_gym/envs/el_4090/envelope_adaptive_2/sl/logs/runs

# 0. 冒烟测试
python legged_gym/legged_gym/scripts/train.py --task=el4090_ea2 --num_envs=4 --max_iterations=1 --headless

# 1. 一键管线（collect → train → eval → export）
python -m $EA2_PKG.pipeline --run-name <name>

# 或分阶段执行：
python -m $EA2_PKG.collect --seeds 1 --num-envs 96          # 数据采集（零动作 rollout）
python -m $EA2_PKG.train --run-name <name> --epochs 50 --patience 10
python -m $EA2_PKG.train --run-name <name> --aux-ks 25,50,100,200,300   # 多档回忆头
python -m $EA2_PKG.eval --ckpt $EA2_RUN/<name>/model.pt     # 闭环评估
python -m $EA2_PKG.export --ckpt $EA2_RUN/<name>/model.pt \
    --out $EA2_RUN/<name>/policy_init.pt --run-name <name>  # 折叠 + 打包为 rsl_rl checkpoint

# 2. 可视化（红/绿点云 + 青色包络）
python legged_gym/legged_gym/scripts/play_ea2.py --task=el4090_ea2 --load_run v3_attitude --checkpoint 0 --num_envs 1

# 3. 可选：从导出权重做 PPO 微调
python legged_gym/legged_gym/scripts/train.py --task=el4090_ea2 --resume --load_run <name> --checkpoint 0
python -m $EA2_PKG.ppo_continue --arm sl_init --ckpt $EA2_RUN/<name>/model.pt --seed 1 --iterations 60 --out p.json
```

### 机器人侧步态策略

```bash
# 训练（68 维条件隐藏策略，主线）
python legged_gym/legged_gym/scripts/train.py --task=el4090_envelop_2 --headless

# 演示（同时自动把策略导出为 TorchScript——cascade 任务的 `policy_1.pt`
# 步态权重即由该导出产物 pin 定）
python legged_gym/legged_gym/scripts/play_envelop_2.py --task=el4090_envelop_2 --num_envs 1

# 旧版 v1（条件可见策略）
python legged_gym/legged_gym/scripts/train.py --task=el4090_envelop --headless
python legged_gym/legged_gym/scripts/play_envelop.py --task=el4090_envelop --num_envs 1
```

### 级联演示与测试

```bash
# 交互演示，cascade_83（权重 pin 在任务包内）
python legged_gym/legged_gym/scripts/play_cascade_83.py --task=el4090_cascade_83 --num_envs 1
#   w/s/a/d/q/e 移动转向，1/2/3 档位，x/空格急停，v 可视化切换，ESC 退出
#   --max_steps N 限步运行（headless 冒烟）

# cascade_68 演示（接口相同；需要 68 维步态权重，待就位）
python legged_gym/legged_gym/scripts/play_cascade_68.py --task=el4090_cascade_68 --num_envs 1

# 测试套件（各含 pytest 契约/感知 + 三个 Isaac env 集成变体）
bash legged_gym/legged_gym/tests/ea2/cascade/run_cascade_tests.sh        # cascade_83
bash legged_gym/legged_gym/tests/ea2/cascade_68/run_cascade68_tests.sh   # cascade_68

# 闭环验收（站立稳定 + vx=1.0 行走）
python legged_gym/legged_gym/tests/ea2/cascade/run_cascade_closed_loop_test.py
python legged_gym/legged_gym/tests/ea2/cascade_68/run_cascade68_closed_loop_test.py
```

## 测试与验收状态

单测为 2026-09-06 在本仓库（干净树 `040340c`）实跑刷新；集成与闭环状态摘自
带日期的任务文档。

| 套件 | 命令 | 结果 |
| --- | --- | --- |
| EA2 全量单测 | `bash legged_gym/legged_gym/tests/ea2/run_ea2_tests.sh` | **287 passed, 10 skipped**（2026-09-06） |
| cascade_83 契约 + 感知 | `run_cascade_tests.sh` 第 [1/4] 步 | **18 passed**（2026-09-06） |
| cascade_68 契约 + 感知 | `run_cascade68_tests.sh` 第 [1/4] 步 | **20 passed**（2026-09-06） |
| cascade_83 env 集成（main / `--ea2-off` / midpoint） | 第 [2–4/4] 步 | PASS（2026-09-01） |
| cascade_83 闭环 | `run_cascade_closed_loop_test.py` | PASS（2026-09-01）——数字见上 |
| cascade_68 env 集成（main / `--ea2-off` / midpoint） | 第 [2–4/4] 步 | PASS（2026-09-04） |
| cascade_68 闭环 | `run_cascade68_closed_loop_test.py` | 待 68 维步态权重 |

EA2 还提供 G0–G5 验收门与诊断探针（oracle 三方归因、通过障碍事件分析、
贴障静止探针等），见
[envelope_adaptive_2/README.md](legged_gym/legged_gym/envs/el_4090/envelope_adaptive_2/README.md) §7。

## 已知局限与路线

- **仅前方感知**：187 通道只覆盖 x ∈ [0.65, 3.65] m、y ∈ [-1, 1] m；侧后方障碍
  只存在于 GRU 记忆中。贴障静止 ≫ 3 s 时包络会缓慢张开（速度混合语料已把
  张开速率压慢约 8 倍并使其饱和）。计划：通道向前方 + 两侧条带重分配；部署
  护栏：ego-motion ≈ 0 时冻结包络。
- **训练域**：EA2 训练域为 vx ∈ [0, 1]、vy ≈ 0、平地 + 柱阵，超出即为 OOD。
  体态回放（`v3_attitude`）已修复真机俯仰伪影；高度/晃动随机化
  （`SwayCfg`/`HeightCfg`）是下一步。
- **地形几何差**：Isaac 侧立柱是 0.1 m heightfield 台阶面，EA2 训练地图是精确
  盒体 mesh（横向命中 ±cm 级偏差）。
- **cascade_68 步态权重待就位**：68 维 SE2 策略仍在训练；就位前 cascade_68
  演示与闭环验收保持锁定。
- **真机部署**：正在推进。

## 仓库结构

```text
el4090_legged_gym/
├── legged_gym/legged_gym/
│   ├── envs/el_4090/               # EL4090 任务包（见任务总览）
│   │   ├── envelope_adaptive/      #   包络模块 v1（融合式，参考保留）
│   │   ├── envelope_adaptive_2/    #   包络模块 v2（EA2，主线）+ sl/ 监督学习管线
│   │   ├── spider_envelop/         #   机器人侧 v1（条件可见）
│   │   ├── spider_envelop_2/       #   机器人侧 v2（条件隐藏 + HAA 网络）
│   │   ├── envelope_cascade_83/    #   级联，冻结 83 维契约（已完成）
│   │   ├── envelope_cascade_68/    #   级联，冻结 68 维契约（步态权重待就位）
│   │   └── ReadMe.md               #   机器人侧 v1/v2 详细文档
│   ├── scripts/                    # train.py、play_ea2.py、play_envelop*.py、play_cascade_*.py 等
│   └── tests/ea2/                  # 单测 + 集成套件、G0–G5 验收门、诊断脚本
├── rsl_rl/                         # vendored rsl_rl（3.3.0）
├── el4090_envelope/                # 独立包络数学与可视化包
└── doc/figures/                    # 本 README 使用的图片与视频
```

## 仓库基础

本仓库是 [legged_gym](https://github.com/leggedrobotics/legged_gym)
的扩展版本，同时作为 [PegasusFlow](https://github.com/MasterYip/PegasusFlow)
的子模块使用。继承自 fork、被包络任务共用的基础设施：

- **rsl_rl 3.3.0 支持**（自 1.0.2 升级），vendored 于 `rsl_rl/`。
- **Nvidia Warp SDF 与 raycasting**：SDF、raycasting 与深度相机集成，供 EA2
  感知路径与 LiDAR 栈使用。
- **Main-rollout 环境架构**（采样类方法用）、受限地形生成与 OBJ 地形支持（参见
  [leggedrobotics/terrain-generator](https://github.com/leggedrobotics/terrain-generator)、
  [MasterYip/blender_robotic_utils](https://github.com/MasterYip/blender_robotic_utils)）、
  gym_visualizer 集成与基准测试工具。
- **`el4090_envelope/`**：包络数学与 Isaac Gym 可视化示例的独立发行版
  （LiDAR 无障碍包络计算、旧版边界滑杆示例），不加载 RL 权重；见
  [el4090_envelope/README.md](el4090_envelope/README.md) 与其
  [可视化指南](el4090_envelope/docs/isaac_gym_examples.md)。
- 早期工作保留的其他 EL4090 环境变体：`spider_nomal`（步态基线）、
  `spider_mammal` / `spider_both`（形态变体）、`safe`（ATACOM 安全层）、
  `pd_gru_lidar`、`data_collect`、`thirdparty`。
