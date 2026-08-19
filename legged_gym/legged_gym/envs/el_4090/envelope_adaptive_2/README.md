# 第二阶段任务书（v2）：基于激光雷达点云记忆网络的六边包络参数预测模块

> v2 依据代码核对结果与最新讨论修订。与 v1 的主要差异：
>
> - 地图改为**全局一张、训练开始时生成后固定**；每 env 的差异只有随机起点/终点；
> - 地形改为 **4×4 地块布局**：每块 16m×16m（总地图 74m×74m，含 5m 外边界），每块地按 **pd_gru_lidar 的随机长方体障碍**参数独立生成，**无实体围墙**，所有长方体轴对齐（不倾斜）；
> - Airy 水平映射修正：**逻辑方位角 0° ↔ Airy 物理通道 30**，分桶选中物理通道 **18~42**（θ=108°~252°），并强制 `airy_mount.py` 自检；
> - LiDAR 噪声改走 `LidarConfig` 的传感器噪声字段，**作用于所有点**（沿用相机传感器原加噪语义，无“仅有效点”分支）；不再使用 `pd_gru_lidar` 手写 domain-rand（注意：当前仓库 LiDAR 路径尚无 `apply_noise()`，需补上，见 §2.2.8 与 §三）；
> - 分桶前**不再做地面过滤**，因此也不存在“干净点云/加噪点云”两份数据；
> - 朝向控制为**切线相对跟踪控制**（位置沿参考路径推进、朝向跟随切线）；
> - 第一阶段**移除所有运动随机量**：速度固定 `1.0 m/s`、`δ_target=0`、无路径横向噪声、无位置/航向晃动、高度固定 `0.52m`；仅保留 LiDAR 点云噪声与地图/起终点随机；
> - 明确 **0.35m 膨胀只保证横向通过**，转弯/端墙处允许最小包络偶发碰撞，由碰撞惩罚给训练信号；
> - 修正 rsl_rl/runner 配置字段归属、BaseTask 必填接口（`max_episode_length`、`infos["time_outs"]`）等与代码冲突的细节。

---

## 一、任务背景与目标

### 1.1 学长已有工作（`spider_envelop_2`）

学长在 `spider_envelop_2` 任务中实现了“用六边包络约束机器人形态”的训练范式：

- **六边包络定义**：由 5 个自由参数描述——`front_width / middle_width / back_width / forward_limit / backward_limit`，经 `apply_env_morphology_priors` 派生为 8 维 condition。
- **包络 → 肢体范围**：condition 经 `HaaRangeNetwork` 蒸馏出 6 条腿的 HAA swing range，机器人腿的摆动被限制在该范围内。
- **Locomotion 训练**：policy observation 为固定 83 维，condition 通过 `_get_structure_condition()` 注入，与 locomotion command `[vx, vy, yaw_rate]` 解耦。
- **奖励侧**：已有 `_reward_haa_range_violation`、`_reward_haa_phase_tracking` 等。

### 1.2 学长预留的接口（含代码核对的精确契约）

- `EnvelopeConditionState.set(values, env_ids, derive_priors=True)`：**当前是 8 维契约**。`condition_names` 含 3 个 prior，`set()` 先对 8 维 clip（prior 范围 `[0,1]`），`derive_priors=True` 时再用前 5 维重算后 3 维。
- `EL_4090_ENVELOP_2.set_envelope_condition(...)`：环境层封装，内部调用 `set()` 并触发 `_refresh_haa_swing_ranges`；**同时还会更新 `embedded_state_default_dof_pos`**（形态目标姿态），M2/M3 接入时必须知道这一行为。
- 5 参数范围与 `apply_env_morphology_priors` 公式的唯一来源是 `spider_envelop/el4090_spider_config.py`；无 Isaac Gym 环境下可用 `legged_gym.utils.envelop.network.haa_swing_range.load_envelope_condition_spec()`（AST 解析）构造 `EnvelopeConditionSpec`，不要手抄数值。
- 解耦保证：condition 由 `EnvelopeConditionState` 独立持有，不读 locomotion commands。

> 本任务要做的，就是**每步用一个感知网络算出 5 维 condition，并转换为 8 维后经 `set()` 喂给下游**；M1 简化环境内部先自行维护 condition 状态。

### 1.3 本任务要解决的问题（`envelope_adaptive_2`）

学长的范式缺少一个闭合环节：**包络参数目前靠随机采样或人工给定，无法根据实际环境自动调整**。本任务训练一个独立的感知网络（GRU），接收机器人坐标系下的前方激光雷达点云序列，直接预测理想的 5 维包络参数。

**总体定位**：

- 与现有 locomotion policy **解耦独立训练**；
- M1 不加载机器人实体，仅移动雷达 + 解析包络；
- 不干预 locomotion 的 PPO 梯度回路。

**本任务不负责的部分**：

- 不重新训练/改动 locomotion policy；
- 不直接控制机器人关节；
- M1 不处理机器人本体与障碍的实际碰撞动力学（碰撞判定针对六边包络 vs 障碍）。

---

## 二、关键技术决策

### 2.1 学习方式：legged_gym PPO

- 使用现有 `OnPolicyRunner` + PPO 框架，policy 类为本仓库 rsl_rl 的 `ActorCriticRecurrent`。
- M1 环境为 **简化 BaseTask**：
  - 无机器人实体；
  - **全局一张** 2D 占据栅格地图（墙体、立柱、窄通道、单侧墙等），**训练开始时生成一次并固定**，不周期重建；
  - 生成后验证地图连通性，避免 A* 无解；
  - 每个 env 每个 episode 只采样自己的起点、终点与 A* 路径（运动参数固定）；
  - 雷达沿该 env 的 A* 参考路径按弧长移动；
  - `step(actions)` 中只更新雷达位姿、生成 LiDAR range image、计算包络奖励。
- 后续 M2/M3 再接已训 locomotion policy 做联合评估。

### 2.2 地图、A* 路径与随机化

#### 2.2.1 地图表示与尺寸

```text
map_size      = 74m × 74m（4×4 块，每块 16m×16m + 5m 外边界；世界坐标 x, y ∈ [-37, 37]）
resolution    = 0.1m
grid_shape    = 740 × 740
occupied      = 1（障碍）
free          = 0（可通行）
world → grid  ix = floor((x + 37.0) / 0.1), iy = floor((y + 37.0) / 0.1)
```

该栅格同时用于：

- A* 路径规划；
- 碰撞惩罚（六边形 vs 占据栅格）；
- 生成 **Warp raycast 网格**（地面平面 + 长方体障碍 box；Isaac 侧静态 actor 仅用于可视化，不是 raycast 几何）。

实现约定：

- **4×4 地块布局**：总地图 74m×74m，均分为 4×4 块（每块 16m×16m），四周 5m 平整边界；
- **每块地独立运行 pd_gru_lidar 的 `pillar_field_terrain`**：随机长/短边长方体、环带放置、AABB + `min_separation` 排斥，参数见 §2.2.2；不同地块随机难度自然形成疏密不同的环境；
- **无实体围墙**：仅把**膨胀后（planning）栅格**的边界标为 blocked，保证 A* 不规划出图，occupancy 与 mesh 边界保持 free；
- **所有长方体轴对齐**（yaw=0），不接受任意倾斜障碍；
- **栅格是权威几何**：先由障碍 footprint 栅格化得到 occupancy，再由同一套参数生成 Warp box 网格，两者必须一致；
- **Warp mesh 必须包含 z=0 地面平面**（建议覆盖地图并外扩 2m）；地面只参与射线，不写入 occupancy、不参与碰撞惩罚——这是“不做地面过滤、地面距离是合法观测”的前提；
- **膨胀自由空间连通性**：最大 8 连通分量须占全部安全格 ≥95%；每 env 的起点/终点只在最大连通分量内采样，保证 A* 有解；
- **地图为全局一张**：所有 env 共享同一 warp mesh，`reset_idx()` 不重建地图，只重采样该 env 的起点/终点/路径状态；
- 生成 Warp mesh 时直接使用长方体基元参数创建水密 box 网格，不要按每个 occupied 栅格生成大量 box，避免 mesh 面数过大。

#### 2.2.2 障碍物参数（完全沿用 pd_gru_lidar）

| 参数 | 值 |
|---|---|
| 每块地数量 | `pillar_count_min=0, pillar_count_max=12`，难度 ∈ {0.5, 0.75, 0.9} |
| 长/宽 | `pillar_size_x = 0.5~4.0m`，`pillar_size_y = 0.5~4.0m`（长边/短边 split 算法） |
| 高度 | `pillar_height = 1.0~2.0m`；`allow_height_variation=True`（最终高度 60%~100%） |
| 间距/放置 | `min_separation=2.2m`，中心净空 `center_clear_radius=3.0m`，放置半径 `spawn_radius=7.5m` |
| 方向 | 全部轴对齐（yaw=0） |

- 碰撞判定只看 2D 投影，不看高度；
- 地图生成验收（不满足则重生成，设最大尝试次数）：
  1. 4×4=16 块地全部为随机长方体地块；
  2. 无实体围墙、全部矩形轴对齐；
  3. 膨胀自由空间最大连通分量 ≥ 95%；
  4. 预采样 8~16 对起点/终点跑 A*，要求全部（或 ≥80%）有解；
  5. 在这些验证路径上，至少 30% 的路径点到最近障碍距离 ∈ `[0.7, 1.5]m`。

#### 2.2.3 六边形包络尺寸与边界

六边形 5 个自由参数的上下界与学长 `spider_envelop` / `spider_envelop_2` 的设计一致：

| 参数 | 下界 | 上界 |
|---|---|---|
| `front_width` | 0.3 | 0.6 |
| `middle_width` | 0.3 | 0.7 |
| `back_width` | 0.3 | 0.6 |
| `forward_limit` | 0.6 | 0.9 |
| `backward_limit` | -0.9 | -0.6 |

几何定义（body 系：+x 向前、+y 向左）：

- 三个 width 是**半宽**，左右对称，完整宽度为其 2 倍；
- 6 个顶点（与 legacy `_build_hex_edges` 一致）：

```text
B=( forward_limit,  front_width)   D=(0,  middle_width)   F=(backward_limit,  back_width)
A=( forward_limit, -front_width)   C=(0, -middle_width)   E=(backward_limit, -back_width)
```

- **最大包络**：长约 `1.8m`（x：`-0.9 ~ 0.9`），宽约 `1.4m`（y：`-0.7 ~ 0.7`）；
- **最小包络**：长约 `1.2m`（x：`-0.6 ~ 0.6`），宽约 `0.6m`（y：`-0.3 ~ 0.3`）。

**碰撞安全口径（v2 明确）**：

- A* 的 0.35m 膨胀只保证**横向通道可通过**（最小半宽 0.3m + 0.05m 余量）；
- 最小包络的**前半长是 0.6m**，因此转弯内角/端墙处最小包络**可能偶发碰撞**，这是有意允许的：碰撞惩罚负责提供“需要收缩”的训练信号；
- 文档不再承诺“最小包络 + 0.05m 永不碰撞”。实现时统计并记录 `min_envelope_collision_free_ratio`（最小包络在 margin=0 下沿路径无碰撞的路径点占比），作为地图生成质量监控，建议目标值 > 90%，实现时调。

#### 2.2.4 A* 路径规划

- 使用 **8 邻接 A\***；
- 代价 = 欧氏距离；启发式 = 欧氏距离；
- **A\* 前先将障碍物按最小包络半宽 + margin 膨胀**：
  - 最小包络半宽 `0.3m` + `0.05m` margin = 膨胀 `0.35m`（3.5 格，实现取 4 格）；
  - 膨胀只做一次（地图固定），并缓存；
- **起点**：episode reset 时在膨胀后的安全栅格中随机采样一个合法出生点，出生朝向对齐路径切线；
- **终点**：随机选取的可行 free 点，要求：
  - 在膨胀后的安全栅格中为 free；
  - 距离最近障碍至少 `0.5m`（按未膨胀的原始 occupancy 计算）；
  - 与起点的 A* 路径长度 ≥ 3m（配置化），保证路径有内容；
- episode 内**不重规划、不换终点**；地图全局固定，因此“当前位置”就是该 env 沿自己路径推进到的位置；
- 生成后处理顺序：A* → line-of-sight 简化 → 按固定间距重采样（`0.2m`）→（路径噪声第一阶段关闭）→ **切线 yaw 按 R_min 平滑** → 输出 `(x, y, yaw)`；
- 最小转弯半径 `R_min ≥ 1.0m`（可配置）：A* 网格的 90° 直角保留在位置几何上（位置仍在膨胀安全栅格内），但**切线 yaw 按 `abs(wrap_to_pi(Δyaw))/Δs ≤ 1/R_min` 平滑**；实际航向角速度由 §2.2.5 的 `ω_max=1.5 rad/s` 限幅，不会因网格直角出现阶跃转向；
- 路径点输出 `(x, y, yaw)`；
- 地图固定后，A* 仍按 env 逐个计算；初始 `num_envs` 建议 `1024~2048`，实现时先 benchmark A* 与 Warp raycast 吞吐再放大；
- 若 Python A* 成为瓶颈：地图固定，可**预计算 K 条候选路径**（或批量/GPU 并行规划），env reset 时采样候选路径，起点改为候选路径起点。

#### 2.2.5 路径速度与朝向（第一阶段：确定性运动）

- 路径速度固定 **`v = 1.0 m/s`**；
- 控制频率 50Hz，每个控制 step 的弧长增量 = `v * dt`；
- **机器人位置**沿 A* 参考路径按弧长推进，保证始终在可行路径上；
- **目标朝向偏置固定 `δ_target = 0`**：heading 始终跟随路径切线；
- **朝向控制**为切线相对跟踪：

```text
s(t+dt)   = s(t) + v·dt
heading(t+dt) = heading(t) + ω·dt
tangent_rate = κ(s)·v
ω_cmd = tangent_rate + k_p · wrap_to_pi(0 − δ_actual)
ω     = clip(ω_cmd, −ω_max, +ω_max)
δ_actual = wrap_to_pi(heading − tangent)
vx = v·cos(δ_actual),  vy = v·sin(δ_actual),  omega = ω
```

- `ω_max = 1.5 rad/s`（可配置）；`k_p = 5.0 1/s`；
- `R_min ≥ 1.0m` 约束参考切线曲率；实际航向角速度由 `ω_max` 限幅；
- 走到路径终点或达到 episode 上限即结束。

#### 2.2.6 路径噪声（第一阶段：关闭）

- `path_noise_amp = 0`：A* → LOS 简化 → 0.2m 重采样后直接作为参考路径，不做横向偏移；
- 路径位置仍全部位于膨胀安全栅格内；
- 后续阶段需要时再开启有界噪声 + 拒绝采样。

#### 2.2.7 运动随机量（第一阶段：全部关闭）

- **位置/航向晃动**：关闭（幅度 0）；
- **高度**：固定 `h = 0.52m`（无目标切换、无过渡、无上下波动）；
- **速度/δ_target**：固定 `1.0 m/s` 与 `0°`；
- 第一阶段仅保留：LiDAR 点云噪声、地图随机长方体、每 env 每 episode 随机起点/终点。

#### 2.2.8 LiDAR 噪声（v2：LidarSensor 噪声字段，作用于所有点）

- M1 开启 LiDAR 噪声，使用 `LidarConfig` 的传感器噪声字段：

```python
enable_sensor_noise      = True
pixel_std_dev_multiplier = 0.02   # 乘性高斯：dist ~ N(dist, 0.02·dist)，作用于所有射线
pixel_dropout_prob       = 0.02   # dropout 概率，作用于所有射线
random_distance_noise    = 0.0    # M1 关闭（原模块不消费该字段）
random_angle_noise       = 0.0    # M1 关闭；M3 若要角度噪声，需新定义语义
```

- **作用范围为所有点（含 no-hit 点），不支持“只对有效点加噪”**：LiDAR 版 `apply_noise()` 按相机传感器原 `apply_noise()` 的语义实现——
  - 乘性高斯：对每条射线的 dist 做 `dist = N(dist, multiplier·dist)`，并令 `points = dir · dist`；
  - dropout：Bernoulli 命中的射线置 `dist = far_plane`、`points = far_plane · dir`（等价“该桶无效”）；
- 为什么“所有点”在这里是安全的：no-hit 点本来就是 `far_plane`（60m），乘性 2% 噪声后仍远大于 5m，不会进入 450 桶；dropout 也只会把点置回“无效”，**不会像 `pd_gru_lidar` 旧实现那样制造 0~0.3m 幽灵点**。聚合层最终仍只接受 `0.2~5.0m` 的真实命中距离；
- **实现方式（禁止照搬相机模块到 xyz 坐标）**：LiDAR 噪声在 **range 域**施加、按原射线方向重建点坐标。相机模块的 `pixels` 是深度标量，直接对 `(x,y,z)` 做独立高斯会引入方向失真、近点符号翻转，且其 dropout 哨兵值（`near_out_of_range_value`）与 LiDAR 的 `far_plane` 无效约定不一致。参考实现如下（torch 侧、raycast 之后调用，不进 Warp graph）：

```python
@torch.no_grad()
def apply_noise(self, pixels: torch.Tensor, dists: torch.Tensor):
    """pixels: (E,1,N,1,3), dists: (E,1,N,1)；返回与输入同形状的加噪结果。
    以传入的 tensor 为权威数据，不依赖 wp.to_torch 是否零拷贝。"""
    if not getattr(self.sensor_cfg, "enable_sensor_noise", False):
        return pixels, dists

    norm = torch.norm(pixels, dim=-1, keepdim=True).clamp_min(1e-6)
    direction = pixels / norm                      # hit 与 no-hit 都是 d·dir，方向可安全复原

    std = float(self.sensor_cfg.pixel_std_dev_multiplier) * dists
    noisy_dists = torch.normal(mean=dists, std=std).clamp_min(0.0)

    p = float(self.sensor_cfg.pixel_dropout_prob)
    if p > 0.0:
        drop = torch.bernoulli(torch.full_like(dists, p)).bool()
        noisy_dists[drop] = self.far_plane          # LiDAR 无效约定，不用相机的 near 哨兵

    noisy_pixels = direction * noisy_dists.unsqueeze(-1)
    pixels.copy_(noisy_pixels)                      # 若为零拷贝视图，同时写回 warp buffer
    dists.copy_(noisy_dists)
    return pixels, dists
```

```python
# LidarSensor.update() 在 raycast 分支（wp.capture_launch 或 wp.launch）结束后：
self.lidar_pixels_tensor = wp.to_torch(self.lidar_warp_tensor)
self.lidar_dist_tensor = wp.to_torch(self.local_dist)
self.lidar_pixels_tensor, self.lidar_dist_tensor = self.apply_noise(
    self.lidar_pixels_tensor, self.lidar_dist_tensor
)
return self.lidar_pixels_tensor, self.lidar_dist_tensor
```

- 训练时 `enable_sensor_noise=True`；play/真机评估用 config 开关显式关闭（建议默认关），避免推理多一个随机源；
- 因为 M1 **不做地面过滤**，也就不需要区分干净点云/加噪点云；碰撞奖励走栅格，不消费点云；
- **代码现状与前置改动**：当前仓库 `LidarSensor.update()` 只执行 raycast，**LiDAR 路径没有任何 `apply_noise()` 实现**；`LidarConfig` 的 4 个噪声字段是“死配置”（git 历史与同源 `Lidar_legged_gym` 包中，唯一成型的 `apply_noise()` 属于相机传感器 `isaacgym_camera_sensor`，其语义就是“对所有 pixel 加噪”）。M1 需按上面契约在 `LidarSensor` 中补一个 LiDAR 版 `apply_noise()`（或在 M1 wrapper 中实现同一函数），并用单测覆盖“no-hit 噪声不进入 450 桶、dropout 置 far_plane、不产生幽灵点”。


### 2.3 雷达安装与分桶（v2：修正后的 Airy 通道口径）

#### 2.3.1 Airy 传感器坐标系与安装

`generate_AIRY()` 的真实坐标系（代码为准）：

```text
x = cosφ·cosθ,  y = cosφ·sinθ,  z = sinφ
φ = 0°~90°（96 线等间隔）,  θ = 0°,6°,…,354°（60 通道）
boresight = 传感器 +z；θ=0° 对应传感器 +x
```

M1 传给 `LidarSensor` 的关键配置（`LidarConfig` 的 `far_plane` 即这里的 `max_range`）：

```python
LidarConfig(
    sensor_type=LidarType.AIRY,
    dt=0.02, update_frequency=10.0,
    max_range=60.0,          # 射线最大测距/far_plane；450 桶另用 5.0m 有效范围
    min_range=0.2,           # 注意：当前 Warp kernel 不消费 min_range，聚合层再过滤
    num_sensors=1,
    horizontal_line_num=60, vertical_line_num=96,  # AIRY 模式下由 pattern 决定，实际 num_vertical_lines=5760
    return_pointcloud=True, pointcloud_in_world_frame=False,
    randomize_placement=False,
    # 噪声字段见 §2.2.8
)
```

安装参数：

```python
offset_pos = [0.0, 0.0, -0.05]               # legacy envelope_adaptive body 安装位置
sensor_offset_rpy = [0.0, π/2 + 0.35, 0.0]   # body→sensor 旋转
```

- **雷达载体高度固定**：M1 载体位姿为 `(x, y, z=0.52m, yaw)`，世界地面 z=0；雷达世界 z=0.47m；
- 参考量级：雷达高 0.47m + boresight 下俯 0.35 rad，boresight **沿射线**约 `0.47/sin(0.35) ≈ 1.37m`（水平投影约 `0.47/tan(0.35) ≈ 1.29m`）处打地——这是不做地面过滤时的合法观测基线；
- 雷达 x/y 与 body 原点重合，六边形以 body 原点绘制，因此点云与包络参考系一致。

用与 `LidarSensor` 完全相同的约定（`quat_from_euler_xyz` → `quat_mul(base_quat, offset_quat)` → kernel 内 `quat_rotate`）换算后：

- boresight（传感器 +z）→ body `(0.939, 0, -0.343)`：**前方偏下 0.35 rad（≈20°）**；
- 传感器 +x → body 后方偏下，因此**传感器 θ=0° 不是逻辑前方**。

#### 2.3.2 水平分桶（v2 已修正）

| 方向 | 方案 |
|---|---|
| 逻辑方位角 | `-72° ~ +72°`，每 6° 一列，共 25 列 |
| 逻辑 0° ↔ Airy 通道 | **物理通道 30（θ=180°）** |
| 选中物理通道 | **18~42**（θ=108°~252°，连续 25 个） |
| 列序 | 列 c = 物理通道 a − 18，对应逻辑方位 `(a−30)·6°` |

- 选定 25×18 条射线在 body 系的覆盖（数值自检期望）：**全部 body-x > 0；方位角约 ±78°；俯仰约 −20° ~ +64°**；
- 这个口径下，“去掉底部 6 线”（传感器 φ 最小的 6 条）对应的是近天空方向，删掉它们是合理的。

#### 2.3.3 垂直分桶

- Airy 垂直通道 `el ∈ [0, 96)`，`φ = el·(90°/95)`；
- 去掉底部 6 线：`el = 0~5`；
- 剩余 `el = 6~95` 共 90 线，每 5 线一桶 → 18 行：`row = (el − 6) // 5`；
- 已知特性：`el=95`（φ=90°）在全部 60 个方位通道上是**同一条方向**，它贡献的距离在 25 列中相同；若它是所在桶的最小距离，则最上一行 25 个值会完全相同。M1 保留（网络可学习），后续可消融移除。

#### 2.3.4 映射表与展平顺序

- `LidarSensor` Airy 输出的展平顺序是 azimuth-major（`theta_grid.flatten()` 的 C 序）：

```text
ray_index i = az·96 + el        # az∈[0,60), el∈[0,96)
```

- 450 维展平顺序固定为 **row-major**：`flat = row·25 + col`，`row∈[0,18)`，`col∈[0,25)`；
- 映射表：对 `i ∈ [0, 5760)`，若 `az ∈ [18,42]` 且 `el ∈ [6,96)`，映射到 `(row, col)` 如上；否则 `-1`；
- 映射表只生成一次，保存为固定常量/文件（`.pt` 或 `.npy`），训练与部署共用；
- **训练/部署通道集合按物理通道索引固定**；若真机 Airy 的通道角度表与 `linspace` 不一致，需用实测角度表重新生成映射，`airy_mount.py` 保留该校准入口。

#### 2.3.5 `airy_mount.py` 强制自检

实现 `airy_mount.py` 时必须包含：

1. 用与 `LidarSensor` 相同的旋转约定，计算 5760 条射线在 body 系的方向；
2. 对 450 桶内的所有射线断言：body-x > 0、body 方位角 ∈ [−80°, +80°]、body 俯仰 ∈ [−25°, +70°]；
3. 保存一张 3D 方向图（或 matplotlib 投影图）到日志，供 play/debug 检查；
4. 验证展平索引公式 `i = az·96 + el` 与映射表内容逐项一致。

### 2.4 Observation

每个控制 step 的输入为：

```text
[range_image_450 / max_range, ego_motion_3]
```

- `range_image`：450 维，每桶取桶内射线的最小距离，空桶填 `max_range`，整体除以 `max_range` 归一化；
- **有效感知距离**：`max_range = 5.0m`（可配置），`r_min = 0.2m`；
  - 有效射线：加噪后 `0.2 ≤ d ≤ 5.0`；
  - 超出范围与 no-hit 射线不参与 min 聚合，对应桶按空桶处理（填 5.0）；
  - `LidarSensor.min_range` 在 Warp kernel 中**不生效**，r_min 必须在聚合层实现；
  - **v2 不做地面过滤**（不再过滤 world z < 0.05）：地面命中是合法距离信息，障碍表现为“地面距离模式上的偏离”，由 GRU 学习；
- `ego_motion`：`[vx, vy, omega]`，按 M1 运动范围归一化：
  - `vx / 1.5`（最大前进速度 1.5 m/s）；
  - `vy / 1.0`（横向速度范围保守取 ±1.0 m/s）；
  - `omega / 1.5`（最大角速度 1.5 rad/s）；
- ego-motion 拼在 GRU **输入**中，GRU 之后不再重复拼接；
- range image 的展平顺序由映射表固定（row-major），训练与部署保持一致。

**频率与数据复用**：

- 控制/PPO step 为 **50Hz**；
- LiDAR 工作频率为 **10Hz**；
- 因此 LiDAR 点云每 **5 个 step** 更新一次，中间 4 个 step 复用同一帧 range image；
- ego-motion 每 step 都更新；
- **LiDAR 触发机制（与 LidarSensor 的全量 env raycast 机制一致）**：沿用 legacy 的**全局 10Hz 时钟**（标量计时器，`decimation = round(1/(dt·10Hz)) = 5`），每第 5 个 step 对**全部 env** 调一次 `LidarSensor.update()`，随后统一刷新 450 维 range image；
- **env reset 的空帧过渡**：某 env 在 **reset 之后、下一次全局扫描之前**，其 range image 置为全 `max_range`（空帧），直到下一次全局扫描刷新（含“reset 与扫描发生在同一步”的情形）；空帧最多持续 4 步，ego-motion 照常更新。这样不会给新 episode 喂上一 episode 的旧地理帧，也不增加 raycast 开销；
- 若后续要求“reset 后第一步就出新鲜帧”，需给 `LidarSensor` 增加 `env_ids` 子集更新能力（kernel/输入数组按子集 launch），作为 M2 可选优化，M1 不要求；
- 传感器位姿在每次 LiDAR 更新时取当时 body 位姿；M1 射线是**瞬时扫描**，不模拟帧内运动畸变（M3 再加）；
- 输入 ego-motion 是 **base 系**运动；传感器装在 `[0,0,-0.05]`（与 base 轴重合），其线速度与 base 线速度一致，网络输入 base 量。

### 2.5 网络结构（v2：修正 rsl_rl 字段归属）

```text
输入 [450 + 3 = 453]
        ↓
单层 GRU（hidden=187）
        ↓
actor: 187 -> 256 -> 128 -> 5
critic: 187 -> 256 -> 128 -> 1
```

- M1 使用 **单层 GRU**；多层 GRU 作为后续消融变体；
- actor/critic MLP head 之间使用 ELU；GRU 内部激活为 PyTorch GRU 默认（tanh/sigmoid）；
- 最终 5 维输出使用 Sigmoid 映射到包络参数合法范围：

```python
norm   = torch.sigmoid(raw_action)
action = low + norm * (high - low)
```

- **Sigmoid/affine 映射在环境侧执行，不放进 actor 网络最后一层**（避免破坏 rsl_rl 高斯 log-prob）；
- 直接复用 rsl_rl `ActorCriticRecurrent`，但**注意 runner 的传参方式**（`on_policy_runner.py::_initialize_policy` 以位置参数传 `num_obs, critic_obs_dim, num_actions`），因此：

```python
# env cfg（观测维度唯一来源）
class env:
    num_observations = 453
    num_actions = 5
    num_privileged_obs = None    # 不用 asymmetric critic；碰撞栅格只进 reward
    episode_length_s = 20.       # max_episode_length = episode_length_s / sim.dt

# PPO cfg（policy_cfg 里绝对不能再写 num_actor_obs / num_critic_obs）
class policy:
    actor_hidden_dims = [256, 128]
    critic_hidden_dims = [256, 128]
    activation = "elu"
    rnn_type = "gru"
    rnn_hidden_dim = 187
    rnn_num_layers = 1
    init_noise_std = 0.3

class runner:
    policy_class_name = "ActorCriticRecurrent"
    algorithm_class_name = "PPO"
    num_steps_per_env = 50      # 50 步=1s≈10 帧 LiDAR；24 步只有约 4 帧，偏短
```

- 本仓库 rsl_rl 已完整支持 recurrent PPO：`RolloutStorage.recurrent_mini_batch_generator` 负责轨迹切分，`PPO.process_env_step` 每步调用 `policy.reset(dones)` 清零 GRU hidden，环境侧无需手动清 hidden；
- `num_steps_per_env` 建议 ≥ 50（实现时可按显存调整，但不建议低于 40）。

### 2.6 Action 与下游接口

- 策略只输出 5 个自由包络参数；
- M1 内部 5→8 转换必须复用下游同一 prior 逻辑：`load_envelope_condition_spec()` + `apply_env_morphology_priors()`（或已验证等价的 legacy `envelope_params_to_condition`），先拼 3 个占位 0，再 derive；
- M1 简化环境内部维护 condition 状态，用于碰撞/势能奖励与可视化；
- M2/M3 接入 locomotion 后，才真正调用 `EL_4090_ENVELOP_2.set_envelope_condition()`：

```python
action_8 = torch.cat([action_5, torch.zeros_like(action_5[:, :3])], dim=-1)
env.set_envelope_condition(action_8, derive_priors=True)
```

- `set()` 的 8 维契约：先对 8 维 clip（拼 0 落在 prior 范围 `[0,1]` 内，安全），`derive_priors=True` 时后 3 维被重算，因此拼 0 不会真正传入“全 0 先验”；
- 注意 `set_envelope_condition()` 还会更新 `embedded_state_default_dof_pos` 并刷新 HAA swing ranges，M2 联合评估时要考虑形态切换的过渡行为；
- 后续可考虑 refactor `EnvelopeConditionState.set()`，让它原生接受 5 维输入。

### 2.7 LiDAR 生成与可视化

- M1 使用 **Isaac Gym 仿真环境 + `LidarSensor` 的 Airy 模式**生成点云（raycast 几何为自建 Warp mesh，不是 Isaac 地形高度场）；
- 地图固定：训练开始时由基元参数生成一次 occupancy → 一次三角网格 → 一次 `wp.Mesh`；
- **Warp 执行路径**：当前仓库没有任何调用方使用 `LidarSensor.capture()`，所有 env 均走 `graph=None` 的直启 kernel；M1 沿用直启即可，地图固定也不存在“换 mesh 后重捕 graph”问题（若未来启用 graph capture，换 mesh 时必须重捕）；
- 环境内部使用映射表把原始点云/距离聚合成 450 维 range image；
- M1 无机器人实体，因此不模拟腿/身自遮挡；M3 接入机器人后再处理；
- 可视化规则（v2 明确）：
  - **红色小球**：通道索引在映射表中且为有效命中（加噪后 `0.2 ≤ d ≤ 5.0`）的点——即真正参与 450 维聚合的点；
  - **绿色小球**：其余有效命中（不在映射表或距离超出有效范围）的点；
  - no-hit、被 dropout 的射线不画；
  - 仅对 `debug_env_ids` 绘制（默认 `[0]`），避免逐 env 画 5760 个球；
  - **六边形包络**：完全复刻 `spider_envelop_2` 的绘制方式——6 条外轮廓 + 1 条中线的**加粗管状线**（`line_radius=0.012, line_samples=8`，青色 `(0,0.85,1)`，`z=0.02`），与点云同一 body 参考系。

### 2.8 Reward

M1 奖励设计：

- **势能项**：鼓励包络展开/变大。M1 默认取归一化参数均值：

  ```text
  potential = (norm(front_width) + norm(middle_width) + norm(back_width)
               + norm(forward_limit) + norm(backward_limit)) / 5
  ```

  其中 `norm` 将参数线性映射到 `[0,1]`；`backward_limit` 先取反再归一化（与 `apply_env_morphology_priors` 物理方向一致）。

- **碰撞惩罚**：六边形与**完整 2D 占据栅格**的侵入，负奖励；不使用点云做碰撞判定；
  - 判定几何为“六边形按每条边外法向平移 `margin=0.05m`”后的扩展轮廓（用 half-plane 外推做精确 offset，不用 legacy 的径向顶点缩放近似）；
  - 碰撞量 = 扩展轮廓覆盖的 occupied 格数与扩展轮廓覆盖格数之比（分母加 `eps` 防除零；配置化可换侵入深度）；
  - 建议默认 `collision = -2.0 · ratio`（可调）；
  - **碰撞栅格只用于 reward，不进 critic/privileged obs**（否则 `num_steps_per_env × num_envs × 14400` 的存储不可行，以 50×2048 为例约 5.9GB）；
  - 碰撞不触发 episode 终止，episode 继续。
- **action rate（小权重）**：惩罚**映射后的 5 维包络参数**逐帧变化：

  ```text
  action_rate = -0.01 · mean((mapped_t − mapped_{t−1})²)     # 建议值，实现时调
  ```

- episode reset 时清零 `last_actions`，避免跨 episode 误罚；
- 建议默认权重：`potential=1.0, collision=-2.0, action_rate=-0.01`，具体值留实现时调；
- **可学性风险（v2 新增）**：碰撞 reward 依赖真实状态与 action，不直接依赖观测，策略存在退化为“输出折中常数包络”的风险。缓解与监控：地图/通道宽度覆盖足够多样、保留 PPO 熵、在 §2.11 增加 action 方差与 clearance-包络相关度指标；M2 再考虑辅助监督——PPO 已预留 `policy.compute_auxiliary_loss` 调用钩子，但 `ActorCriticRecurrent` **没有**该实现，需在 `policy.py` 自定义带 aux head 的 recurrent policy（用 oracle 5 参数做监督）。

### 2.9 训练节奏与 reset

- M1 默认每个 env step 为 **50Hz**（`sim.dt=0.02`，无 decimation）；
- LiDAR 为 **10Hz**，点云/range image 每 5 个 step 更新一次，中间 4 个 step 复用；
- ego-motion 每个 50Hz step 都更新；
- 每个 50Hz step `episode_length_buf += 1`；每步先清 `time_out_buf`，再判断：路径走完或 `episode_length_buf ≥ max_episode_length` 时置 done（`reset_buf`），后者同时置 `time_out_buf=True`；`reset_idx()` 末尾把该 env 的 `reset_buf` 清零；
- **地图全程固定**；episode reset 只重采样该 env 的起点/终点/A* 路径，heading 对齐新路径切线、`δ_actual=0`，速度/高度/晃动为固定值；
- **BaseTask 接口要求（train.py 固定传 `init_at_random_ep_len=True`，漏掉会崩）**：
  - env cfg 提供 `episode_length_s`（如 20s），环境初始化先自建 `self.dt = cfg.sim.dt`，再设 `self.max_episode_length = int(cfg.env.episode_length_s / self.dt)`；
  - `step()` 返回 5 元组 `(obs, privileged_obs, rew, dones, infos)`，且 `infos` 必须含 `"time_outs": self.time_out_buf`（PPO timeout bootstrap 依赖它）；
  - **BaseTask 不会自动 clip**：`step()` 入口先按 `cfg.normalization.clip_actions` 截断 raw actions，返回前按 `cfg.normalization.clip_observations` 截断 obs（参照 `LeggedRobot.step` 的两段 clip）；
  - `infos["episode"]` 里写 §2.11 的指标字典（值必须是 `torch` 标量/0 维 tensor，供 PPO logger 求均值），训练日志直接可见；
- GRU hidden state 随 `dones` 由 `ActorCriticRecurrent`/`policy.reset(dones)` 自动 reset。

### 2.10 ego-motion 与运动随机量

- M1 使用**真值 ego-motion**；
- 第一阶段运动确定：速度 `1.0 m/s`、δ_target `0`、高度 `0.52m`、无晃动/无路径噪声（§2.2.5-§2.2.7）；
- M3 再对 ego-motion 加入估计误差/噪声做域随机化。

### 2.11 评估

- M1 **不加入 oracle 作为评估指标**；
- M1 基础量化指标（写入 `extras["episode"]`，tensorboard 可见）：
  - 平均 reward 是否稳定上升；
  - 平均碰撞惩罚是否下降；
  - 平均包络面积/势能是否在开阔地增大、在窄通道收紧；
  - `min_envelope_collision_free_ratio`（地图生成质量监控）；
  - **policy action 在“开阔/窄通道/单侧墙”三类场景上的方差**，以及“局部 clearance 与输出包络面积的相关度”（防常数策略退化，M1 至少离线统计）；
- 可通过 play 可视化观察策略行为；
- 后续迭代再加入 oracle：当前位姿下最大无碰撞六边形面积，作为 `policy_area / oracle_area`、碰撞率等量化对比。

---

## 三、文件结构与前置改动

新代码放在：

```text
envs/el_4090/envelope_adaptive_2/
  __init__.py
  el_4090_ea2_config.py      # env cfg + PPO cfg（继承 LeggedRobotCfg/CfgPPO 以兼容 parse_sim_params）
  el_4090_ea2_env.py         # BaseTask 子类，M1 无机器人实体
  map_generator.py           # 障碍基元 -> occupancy -> warp mesh（含 z=0 地面）+ 生成验收
  path_planner.py            # A* / LOS 平滑 / 0.2m 重采样 / 路径噪声（第一阶段关闭）/ 切线相对跟踪
  envelope_geometry.py       # 六边形构造、精确 margin offset、栅格碰撞判定
  range_image.py             # Airy 点云 -> 450 维 range image（r_min/max_range 过滤与 min 聚合）
  airy_mount.py              # 安装参数 + 通道口径 18~42 + 映射表 + body 系覆盖自检
  symmetry.py                # 可选：range image 水平镜像（每行 col ↔ 24−col）+ vy/ω 取反、action 不变
  policy.py                  # 默认不需要；M2 加 aux head 时再自定义
  README.md                  # 本文件（实现时随代码迁入本目录）
```

前置小改动（对共享代码）：

1. **`utils/LidarSensor`：补 `apply_noise()`**。当前 `LidarConfig` 噪声字段是死配置、LiDAR 路径没有加噪实现；按 §2.2.8 的契约（相机原语义）在 `update()` 后对**所有射线**施加乘性高斯 + dropout，并加单测（no-hit 噪声不进入 450 桶、dropout 置 far_plane、不产生幽灵点）。`enable_sensor_noise` 默认 `False`，旧任务（`el4090_ea`、`el4090_lidar` 等）不受影响。
2. **`envs/__init__.py`：注册新任务**：

```python
from .el_4090.envelope_adaptive_2.el_4090_ea2_env import EL_4090_EA2
from .el_4090.envelope_adaptive_2.el_4090_ea2_config import El4090EA2Cfg, El4090EA2CfgPPO
task_registry.register("el4090_ea2", EL_4090_EA2, El4090EA2Cfg(), El4090EA2CfgPPO())
```

3. 建议新增最小单测（CPU 可跑）：
   - `tests/test_airy_mount.py`：§2.3.5 的四项断言；
   - `tests/test_range_image.py`：ray 索引公式、映射表 450 形状、空桶填 5.0；
   - `tests/test_envelope_geometry.py`：顶点顺序、半宽语义、margin offset、栅格碰撞；
   - `tests/test_path_planner.py`：A* 连通性、0.35 膨胀、噪声拒绝采样、δ_actual 跟踪。
4. **M1 `create_sim()` 约定**（BaseTask 无机器人）：`add_ground` 建 Isaac 地面、按基元创建静态障碍 actor（仅可视化，放 **env 0** 一份即可）；创建 `num_envs` 个 env handle，**所有 env 共用世界原点（`env_origins=0`，不要按常规网格散开）**——单张 74×74 的 warp mesh 在世界原点，散开会导致 env 射线全部出图；`sensor_pos_tensor / sensor_quat_tensor` 直接用载体世界位姿（不叠加 env_origin），`sensor_pos_tensor / sensor_quat_tensor / mesh_ids` 按 legacy `_init_lidar_sensor` 的模式创建，Warp mesh 是 raycast 权威几何。

- 旧的 `envelope_adaptive/` 规则式实现视为 legacy，不参与新任务。

---

## 四、环节与里程碑

### M1：最小底座

- 简化 BaseTask + **全局固定地图**（74m×74m = 4×4 块，每块 16m×16m；Warp mesh 含 z=0 地面 + pd_gru 随机长方体障碍）+ 每 env A* 路径；
- 起点/终点每 episode 采样；**第一阶段运动确定**：速度 1.0m/s、δ_target=0、高度 0.52m、无路径噪声/无晃动；**切线相对跟踪朝向控制**（ω 限幅 ±1.5 rad/s）；
- 雷达 body 安装 `[0,0,-0.05]`，与六边形同参考系；六边形按 `spider_envelop_2` 加粗管状线绘制；
- Isaac Gym + `LidarSensor` Airy 生成点云，**物理通道 18~42 / 垂直 6~95** 聚合为 450 维 range image，含 `airy_mount.py` 覆盖自检；
- **LidarSensor `apply_noise()` 补齐**：乘性高斯 2% + dropout 2%，作用于所有点（聚合层只收 0.2~5m 有效距离），不做地面过滤；
- 控制 50Hz / LiDAR 10Hz：全局 10Hz 时钟每 5 步扫描全部 env 并刷新 range image，中间 4 步复用；reset 落在两次扫描之间时，该 env 用全 `max_range` 空帧过渡（≤4 步）；
- 红/绿小球可视化采样点与其余点云（仅 debug env ids）；
- 单 GRU(187) + MLP，PPO 训练（`num_steps_per_env≥50`，配置按 §2.5 修正）；
- 奖励：势能 + 碰撞惩罚（完整栅格）+ action rate；补齐 `max_episode_length`/`time_outs` 接口；
- 验证“点云 → 包络”映射可学，并跑 §2.11 的防退化指标。

### M2：信号密度

- 用更密集的**固定地图**配置（重新起训或换种子）训练，更强路径噪声；
- 观察碰撞惩罚变密后策略是否学会收紧；
- 增加 oracle 与 aux head 消融（需在 `policy.py` 自定义 recurrent policy 实现 `compute_auxiliary_loss`，PPO 侧调用钩子已存在）；
- 可选：接入已训 locomotion policy 做联合评估（注意 `set_envelope_condition` 同时更新 embedded target/HAA range）。

### M3：真实化

- 域随机化增强：角度噪声、点丢失增强、帧内运动畸变、晃动增强、ego-motion 噪声；
- 真机 Airy 通道角度表校准（映射表重新生成）与 sim-to-real 验证。
