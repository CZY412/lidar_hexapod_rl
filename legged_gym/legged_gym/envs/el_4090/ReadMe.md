# EL_4090 环境说明

`envs/el_4090` 放置 EL_4090 六足机器人在不同形态/约束设定下的训练环境。当前主要变体包括：

- `spider_nomal`: 基础 spider 形态环境，用作常规基线。
- `spider_mammal`: 偏 mammal 形态的环境变体。
- `spider_both`: 同时覆盖 spider/mammal 形态的环境变体。
- `spider_envelop`: 新增的 envelope 条件环境，用可采样的足端活动范围约束策略。
- `spider_envelop_2`: 将 envelope 保存在策略命令之外，并使用 HAA 范围网络辅助训练；策略不直接观察 envelope。
- `safe`: 带 ATACOM 安全层的环境。
- `thirdparty`: 第三方对称增强/接口相关代码。

本文分别说明 `spider_envelop` 和 `spider_envelop_2`。两者最大的区别是：
`spider_envelop` 把 condition 放入策略观测，而 `spider_envelop_2` 的策略不观察
condition、morphology prior 或 HAA range；这些信息只保存在环境内部。

## Spider Envelop 2：策略和 HAA 网络接口

### 总体数据流

`spider_envelop_2` 将策略接口和 envelope 内部状态分开：

```text
5 维原始 envelope 几何
    -> 内部推导 3 维 morphology prior
    -> 组成 8 维 HAA 网络输入
    -> haa_range.pt 输出六条腿的 HAA 下限/上限
    -> 更新形态中心并计算 HAA 训练奖励

68 维策略观测
    -> locomotion policy
    -> 18 维关节残差动作
    -> q_target = embedded_state_default_dof_pos + 0.35 * action
    -> PD torque
```

Envelope、morphology prior 和 HAA range 都不拼入 68 维策略观测。因此，策略是
在隐藏 envelope 条件下训练的鲁棒策略，而不是显式的 envelope-conditioned policy。

### 策略观测：68 维

策略观测的形状为 `[num_envs, 68]`，布局如下：

| 索引 | 维数 | 内容 | 缩放 |
|---|---:|---|---|
| `[0:3]` | 3 | base linear velocity | `obs_scales.lin_vel` |
| `[3:6]` | 3 | base angular velocity | `obs_scales.ang_vel` |
| `[6:9]` | 3 | projected gravity | 无额外缩放 |
| `[9]` | 1 | `lin_vel_x` command | `command_lin_vel_scale` |
| `[10]` | 1 | `lin_vel_y` command | `command_lin_vel_scale` |
| `[11]` | 1 | `ang_vel_yaw` command | `command_ang_vel_scale` |
| `[12:30]` | 18 | `dof_pos - embedded_state_default_dof_pos` | `obs_scales.dof_pos` |
| `[30:48]` | 18 | DOF velocity | `obs_scales.dof_vel` |
| `[48:66]` | 18 | 当前已施加的上一帧策略动作 | 无额外缩放 |
| `[66]` | 1 | `sin(gait_phase)` | `obs_scales.gait_phase` |
| `[67]` | 1 | `cos(gait_phase)` | `obs_scales.gait_phase` |

其中明确不包含：

- 5 维 envelope 几何；
- 3 维 morphology prior；
- 6 个 HAA range center；
- 6 个 HAA half range。

关节相关观测和动作使用 Isaac Gym 中的 DOF 顺序。当前 EL4090 顺序是：

```text
LB_HAA, LB_HFE, LB_KFE,
LF_HAA, LF_HFE, LF_KFE,
LM_HAA, LM_HFE, LM_KFE,
RB_HAA, RB_HFE, RB_KFE,
RF_HAA, RF_HFE, RF_KFE,
RM_HAA, RM_HFE, RM_KFE
```

### 策略输出：18 维

策略输出形状为 `[num_envs, 18]`，每个值是对应关节的目标位置残差，顺序与上面的
18 个 DOF 完全一致。它不是绝对关节角，也不是 torque。当前 P 控制逻辑为：

```text
action_scaled = 0.35 * action
q_target = embedded_state_default_dof_pos + action_scaled
torque = Kp * (q_target - q) - Kd * dq
torque = clip(torque, -torque_limit, torque_limit)
```

`embedded_state_default_dof_pos` 由三个 morphology prior 在 spider 和 mammal 默认姿态
之间插值得到。因此，虽然策略看不到 morphology prior，内部控制中心仍会随 envelope
变化。策略观测使用相对于这个控制中心的关节误差，保证训练和部署使用同一参考系。

### `haa_range.pt` 输入：8 维

网络输入形状为 `[batch, 8]`，顺序必须严格保持为：

| 索引 | 名称 | 当前范围 | 来源 |
|---|---|---:|---|
| `0` | `front_width` | `[0.3, 0.6]` | 原始 envelope |
| `1` | `middle_width` | `[0.3, 0.7]` | 原始 envelope |
| `2` | `back_width` | `[0.3, 0.6]` | 原始 envelope |
| `3` | `forward_limit` | `[0.6, 0.9]` | 原始 envelope |
| `4` | `backward_limit` | `[-0.9, -0.6]` | 原始 envelope |
| `5` | `morphology_front_prior` | `[0, 1]` | 由前 5 维推导 |
| `6` | `morphology_middle_prior` | `[0, 1]` | 由前 5 维推导 |
| `7` | `morphology_back_prior` | `[0, 1]` | 由前 5 维推导 |

外部感知或规划模块只需要提供前 5 个独立几何量。后三个 prior 必须使用训练时相同的
`apply_env_morphology_priors()` 逻辑计算，不能在部署时独立随机生成。

### `haa_range.pt` 当前输出：每腿上下界

当前 checkpoint 由 `HaaRangeNetwork.from_checkpoint()` 加载，对外输出形状为
`[batch, 6, 2]`：

```text
output[:, leg, 0] = HAA lower bound
output[:, leg, 1] = HAA upper bound
```

checkpoint 的六腿输出顺序为：

```text
RF, RM, RB, LF, LM, LB
```

环境随后使用索引 `[5, 3, 4, 2, 0, 1]` 转换成仿真 HAA 顺序：

```text
LB, LF, LM, RB, RF, RM
```

网络最后一层实际产生 12 个 raw values，并在 `forward()` 内部构造合法范围：

```text
center = joint_lower + sigmoid(raw_center) * (joint_upper - joint_lower)
max_half = min(center - joint_lower, joint_upper - center)
half_range = sigmoid(raw_half_range) * max_half
lower = center - half_range
upper = center + half_range
```

因此当前文件的环境输出契约是 `[lower, upper]`，不是直接的
`[center, half_range]`。

### 替换为 center/half-range 输出的网络

新网络可以直接输出 `[center, half_range]`，形状仍为 `[batch, 6, 2]`，但不能只覆盖
现有 `.pt` 文件。加载或适配层必须先将它转换成环境统一使用的上下界：

```python
center = prediction[..., 0].clamp(joint_lower, joint_upper)
half_range = prediction[..., 1].clamp_min(0.0)
max_half_range = torch.minimum(
    center - joint_lower,
    joint_upper - center,
).clamp_min(0.0)
half_range = torch.minimum(half_range, max_half_range)

lower = center - half_range
upper = center + half_range
ranges = torch.stack((lower, upper), dim=-1)
```

建议新 checkpoint 显式保存以下元数据，避免把两种输出语义混用：

```python
{
    "output_format": "center_half_range",
    "condition_names": [...8 个输入名称...],
    "leg_names": ["RF", "RM", "RB", "LF", "LM", "LB"],
}
```

环境内部可以继续统一保存 `[lower, upper]`。这样 analytic、Monte Carlo 和 network
三种 estimator 保持相同的下游接口，HAA 奖励代码无需分支。

### HAA range 在训练和部署中的用途

训练阶段，包络改变时才重新运行 HAA estimator。输出用于：

1. `haa_range_violation`：惩罚 HAA 关节位置超出 `[lower, upper]`；
2. `haa_phase_tracking`：由范围中心和半范围构造六腿相位目标。

HAA range 不进入策略观测，也不直接裁剪策略动作。当前部署只运行 locomotion policy
和 P/PD 控制时，可以不加载 `haa_range.pt`；如果部署端还需要 HAA 硬限幅、安全监控
或在线形态约束，则应保留该网络，并在控制安全层中使用它的范围输出。

## Spider Envelop 目标

`spider_envelop` 的核心目标是让策略学会在不同足端活动范围下运动。环境每隔一段时间采样一组 envelope condition，表示机器人身体坐标系下允许足端落点出现的平面范围。

策略会在 observation 中直接看到这些 condition，因此它不是只学一个固定形态，而是学一个条件策略：

```text
policy(obs, velocity command, envelope condition) -> action
```

如果足端超出当前 envelope，环境通过 `envelope_constraint` 奖励项给惩罚。这样可以训练策略根据不同结构范围自动调整步态和腿部摆动。

## Command 和 Condition 布局

`spider_envelop` 中 `commands.num_commands = 4 + 8`，总共 12 维：

```text
commands[0]  lin_vel_x
commands[1]  lin_vel_y
commands[2]  ang_vel_yaw
commands[3]  heading
commands[4:12] envelope condition
```

其中 condition 的顺序由 `condition_names` 定义：

```python
condition_names = [
    "front_width",
    "middle_width",
    "back_width",
    "forward_limit",
    "backward_limit",
    "morphology_front_prior",
    "morphology_middle_prior",
    "morphology_back_prior",
]
```

前 5 个是 envelope 的几何边界，后 3 个是由 envelope 推导出的形态先验。

## Envelope 几何定义

Envelope 是机器人 base yaw 坐标系下的 2D 足端范围。代码中会把 condition 转成 6 个边界点：

```text
(forward_limit,  front_width)
(0,              middle_width)
(backward_limit, back_width)
(backward_limit, -back_width)
(0,              -middle_width)
(forward_limit,  -front_width)
```

也就是说：

- `front_width`: 身体前方区域的左右半宽。
- `middle_width`: 身体中部区域的左右半宽。
- `back_width`: 身体后方区域的左右半宽。
- `forward_limit`: 足端允许到达的最前方 x 边界。
- `backward_limit`: 足端允许到达的最后方 x 边界，通常为负值。

当前默认范围在 `spider_envelop/el4090_spider_config.py` 中：

```python
front_width = [0.3, 0.6]
middle_width = [0.3, 0.7]
back_width = [0.3, 0.6]
forward_limit = [0.6, 0.9]
backward_limit = [-0.9, -0.6]
```

环境在计算超界惩罚时，会把足端位置转换到 base yaw 坐标系下，再检查足端是否落在这个由前/中/后三段线性插值得到的范围内。

## Morphology Prior 设计

`morphology_front_prior`、`morphology_middle_prior`、`morphology_back_prior` 不直接随机采样，而是由前 5 个 envelope 几何参数计算得到。

它们的取值范围是 `[0, 1]`：

- `0` 更偏 spider 默认姿态。
- `1` 更偏 mammal 默认姿态。

默认模式是：

```python
morphology_prior_mode = "directional_ratio"
```

这个模式会把横向宽度和前后 reach 的关系转成形态先验。直觉上：

- 横向范围越宽，越偏 spider。
- 前后 reach 越强，越偏 mammal。
- middle 部分主要由 `middle_width` 决定。

这些 prior 会用于两个地方：

1. **默认关节目标插值**

   `embedded_state_default_dof_pos` 会根据 prior 在 spider 默认关节角和 mammal 默认关节角之间插值：

   ```text
   target = spider_default + prior * (mammal_default - spider_default)
   ```

   front/middle/back 三组腿分别使用对应的 prior。

2. **机身高度目标插值**

   base height target 会在 spider 高度和 mammal 高度之间插值：

   ```python
   base_height_spider_target = 0.53
   base_height_mammal_target = 0.64
   ```

这样 envelope 不只是一个惩罚边界，也会影响机器人应该采取的身体形态。

## Observation 设计

`spider_envelop` 的基础观测维度为 74，开启 LiDAR 后总维度为：

```python
num_observations = 74 + 11 * 17
```

基础观测由以下部分拼接：

```text
base_lin_vel                 3
base_ang_vel                 3
projected_gravity            3
lin_vel_x command            1
lin_vel_y command            1
ang_vel_yaw command          1
envelope condition           8
dof_pos - condition target   18
dof_vel                      18
last actions                 18
```

注意：condition 在 observation 中直接使用 `commands[:, condition_start_idx:condition_end_idx]` 输入。因为 condition 在采样时已经按配置范围生成，所以这里不再额外裁剪。

噪声配置中 commands 段被置为 0 噪声，包含速度命令和 envelope condition，避免策略看到被扰动的目标条件。

## Condition 采样流程

正常训练时，`_resample_commands(env_ids)` 会周期性重新采样速度命令和 envelope condition。

Condition 采样步骤：

1. 在 `[0, 1]` 中随机采样 8 维向量。
2. 映射到 `condition_low ~ condition_high`。
3. 用 `_set_morphology_prior_from_envelope()` 根据前 5 个几何参数重算后 3 个 morphology prior。
4. 写入 `commands[:, 4:12]`。
5. 根据新 condition 更新 `embedded_state_default_dof_pos`。

因此，配置里的 `morphology_*_prior` 范围用于声明合法范围，但正常情况下它们会被 envelope 几何覆盖，而不是独立随机决定。

## Envelope Reward

新增奖励项：

```python
envelope_constraint = -10.0
```

对应实现是 `_reward_envelope_constraint()`。它会：

1. 取当前 condition 并生成 envelope 边界。
2. 将每个足端位置转换到 base yaw 坐标系。
3. 根据足端 x 所处的前半区/后半区，线性插值得到当前 x 下的左右边界。
4. 对超出 x/y 边界的距离平方求平均。
5. 只有移动命令超过 `envelope_constraint_min_command` 时才启用惩罚。

相关配置：

```python
envelope_constraint_margin = 0.0
envelope_constraint_min_command = 0.15
```

`margin` 可以扩大或收紧判定边界；`min_command` 用来避免站立时也强行惩罚足端范围。

## Envelope 可视化

可视化开关在 commands 配置中：

```python
envelope_debug_viz = True
envelope_debug_env_ids = [0]
envelope_debug_ground_z_offset = 0.02
envelope_debug_color = (0.0, 0.85, 1.0)
envelope_debug_line_radius = 0.012
envelope_debug_line_samples = 8
```

当前可视化会把 envelope 画成贴近地面的 2D 轮廓线：

- 跟随机器人 `x/y` 位置移动。
- 跟随机器人 yaw 旋转。
- 不再画高度柱体，只画地面 footprint。
- `ground_z_offset` 默认 2 cm，用来避免线和地面重叠闪烁。

这个视图主要用于观察足端是否越过当前 condition 对应的 envelope 范围。

## Morphology Reachability Test

`morphology_reachability_test` 是 debug/test 功能，不是正常训练采样逻辑。

```python
morphology_reachability_test = False
morphology_reachability_test_mode = "corners"
morphology_reachability_resample_steps = 600
morphology_reachability_print_interval = 100
```

打开后，环境会按固定模式采样 condition，并打印指定 env 的可达性状态。`morphology_reachability_test_mode` 有三种：

- `"center"`: 所有 condition 取范围中点。
- `"random"`: 每次随机采样 condition。
- `"corners"`: 对非 `morphology_` 的几何 condition 取 low/high 组合，用来测试参数空间角点。

`corners` 适合检查最窄、最宽、最靠前、最靠后的极端 envelope 下，默认形态目标和足端范围是否合理。

## 与其他变体的关系

`spider_envelop` 继承自 `ElSpider`，不是 ATACOM safe 环境。它的约束来自 reward 和 condition，而不是动作投影安全层。

和普通 spider 环境相比，它主要新增：

- command 从原来的速度/heading 扩展为速度 + envelope condition。
- observation 增加 8 维 condition。
- DOF position observation 使用 `dof_pos - embedded_state_default_dof_pos`，其中默认姿态由 condition 决定。
- reward 增加 envelope footprint 超界惩罚。
- debug viewer 增加跟随机器人移动的地面 envelope 轮廓。
- 提供 morphology reachability test 用于检查 condition 设计是否可达。

`safe` 环境仍然用于 ATACOM 安全层实验；`spider_envelop` 更适合研究“给策略一个结构/包络条件，让策略在该条件下学习运动”。
