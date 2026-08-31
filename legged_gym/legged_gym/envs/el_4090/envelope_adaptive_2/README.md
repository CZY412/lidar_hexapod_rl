# EA2：基于激光雷达点云记忆网络的六边包络参数预测（v3）

> 本文档取代旧版 v2 任务书，与当前代码一一对应。核心目标不变：训练一个 GRU
> 感知网络，输入机器人前方 LiDAR 点云序列，输出"恰到好处"的六边包络参数——
> 在障碍约束下尽量展开。仓库内有两套训练框架：**监督学习（`sl/`，主线）** 与
> **PPO（rsl_rl，微调阶段）**，共用同一个简化环境底座。

---

## 1. 任务定义

- **观测（190 维）**：187 通道固定地面网格 range image（11 行 × 17 列，覆盖
  体坐标前方 x∈[0.65, 3.65] m、y∈[-1, 1] m，除以 max_range=3.2 归一化，无命中
  置 1.0）+ 3 维 ego-motion `[vx/1.5, vy/1.0, ω/1.5]`。
- **动作（5 维）**：六边包络参数 `[front_width, middle_width, back_width,
  forward_limit, backward_limit]`，合同边界
  `low=[0.3,0.3,0.3,0.6,-0.9]`、`high=[0.6,0.7,0.6,0.9,-0.6]`（边界由
  spider_envelop 配置冻结，`test_contracts.py` 钉死）。
- **归一化约定**：内部统一使用归一化 extent `s ∈ [0,1]`（0=最小包络 1.2×0.6 m，
  1=最大包络 1.8×1.4 m；`backward_limit` 因物理方向相反，符号取反）。
- **动作映射（全仓库唯一真相）**：env 把原始动作 a 映射为
  `params = clamp(mid + a·scale, low, high)`，即
  `s = 0.5 + k·sign·a`，其中 `k = soft_dof_pos_limit/(2·action_max)`
  （0.95/8 = 0.11875），`sign = [1,1,1,1,-1]`。
  **k 由 `sl/sl_config.env_action_scale()` 从活配置惰性派生**——导出折叠与
  评估逆映射都消费它，禁止再硬编码 0.1125。

## 2. 共享环境底座（`el_4090_ea2_env.py`）

两套框架共用同一个简化 `BaseTask`（无机器人实体），其最关键的特性是
**动作不影响状态转移**——运动纯脚本化，这使离线监督在数学上等于部署行为：

- **地图**：74 m×74 m 全局一张（4×4 块 16 m 柱状障碍地块 + 5 m 边界，0.1 m
  栅格），由 `cfg.seed` 决定布局；Warp mesh 为 raycast 权威几何；距离场一次
  预计算。无实体围墙，仅规划栅格边界阻塞。
- **路径**：A*（0.4 m 膨胀栅格）→ LOS 简化 → 0.2 m 重采样 → 切线平滑；到点
  soft-replan、转角 turn-in-place；速度 1 m/s，`ω_max=1.5`，episode 45 s。
- **LiDAR**：完整 Airy（900 方位 × 96 俯仰）中预选 187 通道
  （`selected_airy_channels.pt`，训练/部署共用）；10 Hz 刷新，50 Hz 控制每步
  消费；乘性高斯 2% + dropout 2% 噪声作用于全部射线。
- **Oracle（监督目标）**：`envelope_oracle.py` 从全局距离场直接计算理论目标
  包络——axis 模式轴向 march（margin 0.20）→ 扩展侧软上限 → active-set 联合
  收缩（保证 24 个边界采样点无违约）→ `RateLimitedOracle` 快收慢放平滑
  （收 2.0 m/s、放 0.5 m/s、冷却 0.2 s）。平滑输出 `prev_s` 即 SL 回归目标与
  PPO 奖励参考。
- **碰撞语义**：违规 = 34 个采样点对距离场的 clearance < margin(0.10 m)
  （软坡宽 0.10 m）。**floor-pinned 帧**（最小包络也放不下，约为帧数的 1.8%）
  几何不可行，README v2 即豁免，所有归因以此为底线。

## 3. 文件结构

```text
envelope_adaptive_2/
├── el_4090_ea2_env.py        # EL_4090_EA2 简化 BaseTask（两框架共用底座）
├── el_4090_ea2_config.py     # env cfg + PPO cfg（LeggedRobotCfg 血统）
├── _contracts.py             # 冻结契约：常数/数据类/函数签名（勿改）
├── map_generator.py          # 柱状场地形 → occupancy → Warp mesh + 距离场
├── path_planner.py           # A* / LOS / 重采样 / 路径噪声 / 切线平滑
├── path_batch.py             # 路径的 GPU padded 镜像 + 批量插值查询
├── path_parallel.py          # fork 多进程批量规划 worker
├── envelope_geometry.py      # 六边形构造 / 精确 margin offset / 栅格碰撞
├── envelope_oracle.py        # 几何 oracle（axis march + active-set 收缩）
├── target_smoother.py        # RateLimitedOracle 快收慢放目标平滑
├── range_image.py            # 187 通道 range image 构建/归一化
├── airy_mount.py             # Airy 方向生成 + 187 通道选择 + 自检
├── selected_airy_channels.pt # 固定 187 通道表（训练/部署共用）
├── README.md                 # 本文档
├── __init__.py               # 任务注册 el4090_ea2
└── sl/                       # ── 监督学习框架（主线）──
    ├── __init__.py           # 管线概览 + G0-G5 门的环境变量约定
    ├── sl_config.py          # SL 全部配置 + env_action_scale() 动作标度派生
    ├── dataset.py            # 采集（零动作 rollout）/ 数据类 / 滑窗数据集
    ├── model.py              # EnvelopeNet（与 rsl_rl actor 半边逐键同构）
    ├── train.py              # 训练循环：MSE + 回忆辅助头 + 可微安全损失
    ├── evaluate.py           # 闭环评估（stateful/window + 基线对照）
    ├── export.py             # 权重移植 + 动作折叠 + rsl_rl checkpoint 打包
    ├── scripts/
    │   ├── collect.py        # 采集 CLI（每 seed 一个进程）
    │   ├── train.py          # 训练 CLI（config 为默认，CLI 仅覆盖）
    │   ├── eval.py           # 闭环评估 CLI（每 seed 一个子进程）
    │   ├── export.py         # 导出 CLI（校验 14 项 + 写 PPO 可达副本）
    │   ├── pipeline.py       # 一键串联五阶段（每阶段独立进程）
    │   └── ppo_continue.py   # PPO 三臂对照（scratch/sl_init/sl_init_cross）
    └── logs/
        ├── data/             # map_seed<N>.pt 采集数据（gitignored）
        └── runs/<name>/      # model.pt + model_metrics.json + sl_config.json
                              # + closed_loop*.json + ppo_*.json + policy_init.pt
```

PPO 侧共享：`rsl_rl/`（ActorCriticRecurrent 等）、`legged_gym/scripts/train.py`
（PPO 训练入口）、`legged_gym/scripts/play_ea2.py`（点云可视化 play）。

## 4. 监督学习框架（主线）

### 4.1 原理

EA2 的状态转移与动作无关，因此**零动作 rollout 采集的数据就是部署时的状态
分布**，离线指标即部署真值。奖励 `r = -3·MSE(a, oracle)` 即时、确定、无跨步
credit assignment，直接监督严格优于用 PPO 采样同一信号，且秒级 vs 小时级。

### 4.2 管线五阶段

```text
collect → train → eval → export → ppo_continue(可选)
```

1. **collect**：零动作 rollout，每控制步存一帧
   `(obs, target=smoother.prev_s, done, heading, pos)`，每 seed 一个
   `map_seed<N>.pt`（meta 记录 oracle 配置/soft 上限/warmup 供审计溯源）。
   ⚠ Isaac 单进程单环境约束：每 seed 必须独立进程。
2. **train**：滑窗（seq_len=200 帧=4s，stride=10，跨 done 丢弃，按 env 划分
   train/val 防泄漏），MSE + 两个辅助项（见 4.3），val-R² 早停。
3. **eval**：闭环回灌 env，对照 zero-action / 最优常数包络 / stateful /
   window 四种策略的 reward、MSE、碰撞率、面积。**stateful（逐帧带隐状态）
   为推荐部署模式**，实测远优于复现训练窗口的 window 模式。
4. **export**：10 个张量按键移植进 `ActorCriticRecurrent`，并**把 s→a 仿射折
   叠进 actor 末层**（`a = sign·(s-0.5)/k`）——没有这一步，play 的包络会被
   压死在中点附近。critic 保持随机初始化；14 项校验全过才落盘。
5. **ppo_continue**：PPO 微调三臂对照，验证 SL 权重起点价值（G5）。

### 4.3 定版训练配方（C 方案，`sl/logs/runs/baseline_50hz_recall75_safe`）

```python
cfg.model.aux_ks    = [75]      # 回忆辅助头：h_t 重建 s_{t-75}（1.5s ≈ 通过障碍的不可见尾巴）
cfg.model.aux_mode  = "recall"  # 纯记忆压力，无法被当前视野外推满足
cfg.train.aux_beta  = 0.5
cfg.train.safe_lambda = 1.0     # 可微碰撞损失：双线性距离场采样、floor-pinned 掩零
```

```text
L = MSE(ŝ_t, s_t) + 0.5·recall_loss + 1.0·safe_loss
```

设计要点（均为实验结论）：

- **回忆头**在 h 中显式建立"记住滑过视野的障碍"的压力（探针误差 0.039→0.023）；
  正向预测头（A 组）因含不可约的猜测成分而全面劣于回忆（B 组），已否决。
- **安全损失**在策略自身预测真实几何违规时给出直接梯度；**必须用双线性距离场
  采样**（env 奖励的最近邻查表对参数梯度恒为零，不可作 SL 损失）。
- **MSE 锚必须保留**：防止安全损失把包络压向最小（clearance 塌缩）。

### 4.4 当前结果（4 图池化，部署节奏 50Hz）

| 指标 | 对照（纯 MSE） | **定版 C** |
|---|---|---|
| val R² | 0.807 | 0.798 |
| policy 不安全帧率（含 1.8% floor-pinned） | 5.10% | **3.64%** |
| pass-by 尾部越界事件率 | 7.5% | 6.4% |
| policy 平均最小 clearance | 0.0949 m | 0.0964 m（oracle 0.0982，无塌缩） |

归因结论：oracle 目标在几何可行帧上零碰撞（其不安全率 ≡ floor-pinned 率），
平滑器零额外碰撞；全部残余来自网络。**残余的 ~6.4% 通过事件受制于观测边界**
——187 通道只看前方，侧后方依赖 GRU 记忆；结构性解法是 187 通道重分配
（前方网格 + 两侧条带，obs 维度不变），见 §8。

## 5. PPO 框架（微调阶段）

- **网络**：rsl_rl `ActorCriticRecurrent`，单层 GRU(187) + actor/critic 各
  [256,128] MLP，`num_observations=190`，`num_actions=5`。SL 导出的权重即其
  初始化（`play_ea2` / `train.py --resume` 直接加载）。
- **奖励**（`El4090EA2Cfg.rewards.scales`）：`oracle_mse=-3.0`（主导）、
  `envelope_limits=-0.8`（软限位）、`action_rate=-0.01`；
  `potential=0`、`collision=0`（当前作为遥测，供后续启用真实碰撞惩罚）。
- **PPO 超参**（LeggedRobotCfgPPO 默认 + EA2 覆盖）：lr 1e-3 adaptive、
  γ 0.99、λ 0.95、clip 0.2、entropy 0.01、`num_steps_per_env=100`、
  `max_iterations=3000`。
- **已知风险**：从 SL 初始化起步的 PPO 前 60 轮会出现策略退化（随机 critic +
  std=0.5 探索），但起点显著优于 scratch（-0.14 vs -0.69）。微调时建议低
  std / 低 lr 起步。
- **入口**：`legged_gym/scripts/train.py --task=el4090_ea2`；可视化
  `legged_gym/scripts/play_ea2.py`（1 env、关噪声、红/绿点云 + 青色包络）。

## 6. 常用命令速查

环境：`conda activate el4090`；gymtorch JIT 需要 ninja 在 PATH
（`PATH=/home/t3chichi/anaconda3/envs/el4090/bin:$PATH`）。

```bash
PKG=legged_gym.envs.el_4090.envelope_adaptive_2.sl.scripts

# 1. 采集（每 seed 独立进程！Isaac 单进程单环境）
python -m $PKG.collect --seeds 1 --num-envs 96 --num-steps 1400
python -m $PKG.collect --seeds 21 --pillar-counts 28 ...   # 稠密图

# 2. 训练（配置即默认；CLI 仅显式覆盖）
python -m $PKG.train --run-name <name> --epochs 50 --patience 10

# 3. 闭环评估
python -m $PKG.eval --ckpt sl/logs/runs/<name>/model.pt

# 4. 导出为 rsl_rl 策略（--run-name 同步写 logs/el4090_ea2/<name>/model_0.pt）
python -m $PKG.export --ckpt sl/logs/runs/<name>/model.pt \
    --out sl/logs/runs/<name>/policy_init.pt --run-name <name>

# 5. 可视化
python legged_gym/legged_gym/scripts/play_ea2.py \
    --task=el4090_ea2 --load_run <name> --checkpoint 0 --num_envs 1

python legged_gym/legged_gym/scripts/play_ea2.py \
    --task=el4090_ea2 --load_run recall75_safe --checkpoint 0 --num_envs 1

python legged_gym/legged_gym/scripts/play_ea2.py \
    --task=el4090_ea2 --load_run v2_multik --checkpoint 0 --num_envs 1

# 6. PPO 微调（从导出起点 resume）与三臂对照
python legged_gym/scripts/train.py --task=el4090_ea2 --resume --load_run <name> --checkpoint 0
python -m $PKG.ppo_continue --arm sl_init --ckpt <model.pt> --seed 1 --iterations 60 --out p.json

# 一键管线（可 --skip-collect / --stages eval,export）
python -m $PKG.pipeline --run-name <name>
```

## 7. 测试与诊断

```bash
python -m pytest legged_gym/tests/ea2 -q     # 单测全量（无外部依赖）
# G0-G5 验收门（指向真实产物时自动启用）：
export EA2_SL_DATA_DIR=<...>/sl/logs/data          # G0 数据契约
export EA2_SL_METRICS=<...>/runs/<name>/model_metrics.json   # G2 训练指标
export EA2_SL_CKPT=<...>/runs/<name>/model.pt      # G4 导出移植
export EA2_SL_EVAL=<...>/runs/<name>/closed_loop.json        # G3 闭环行为
export EA2_SL_PPO_DIR=<...>/runs/<name>            # G5 PPO 三臂
```

诊断脚本（`tests/ea2/`，非 pytest 收集，直接运行）：

- `validate_sl_vs_oracle_audit.py`：三方归因（raw oracle / 平滑目标 / 策略）
  的碰撞率、收缩幅度、滞后、可观测性；`--ood-seed N` 现采未见地图做泛化
  检查。因为动作不影响状态，**离线审计 = 部署精确复现**。
- `validate_pass_by_expansion.py`：通过障碍场景的对齐事件分析（提前扩张 /
  尾部越界率），记忆类改进的验收仪器。
- `validate_sl_vs_oracle_snapshots.py`：按失败类别出俯视 PNG
  （策略青 / oracle 橙 / 最小包络灰虚线 / 最大包络绿点线）。

## 8. 关键约定、陷阱与路线

- **动作折叠**：SL 网络输出 s∈[0,1]，env 消费 raw action——导出必须折叠
  （`export.export` 默认开启）；`verify_export` 端到端校验，不可跳过。
- **标度单一来源**：`env_action_scale()` 从活配置派生；`soft_dof_pos_limit`
  变更会自动传导，但历史快照数据的 target 上限（0.95=soft 0.9 时代）要与
  meta 对照。
- **节奏一致**：采集/训练/评估/部署全部按 50Hz 控制（`lidar_decimation=1`）。
  历史上 10Hz 训练 + 50Hz 部署造成 6 倍 MSE 失配，勿回退。
- **Isaac 约束**：单进程只能建一个 `EL_4090_EA2`（第二次构造 segfault）；
  多 seed/多阶段一律子进程。
- **oracle 残余边界**：floor-pinned 帧（~1.8%）几何不可行，属设计豁免
  （README v2 §2.2.3）；残余 pass-by 事件受 187 通道前向视野限制。
- **后续路线**：① 187 通道重分配（前方 + 两侧条带，obs 维度不变，治侧后
  不可观测）；② PPO 微调启用真实碰撞惩罚（当前权重为 0）并压探索噪声；
  ③ map_generator 实现 contracts 中已定义的 corridor/side_walls tile，
  补长障碍训练分布。
