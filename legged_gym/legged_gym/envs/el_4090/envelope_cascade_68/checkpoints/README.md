# checkpoints/ — cascade_68 的全部策略权重（pin 进 git，冻结 68 维契约）

演示采取**策略集中管理**：权重都收在本目录，运行不依赖 `logs/` 或其他任务目录。

**冻结契约**：本任务与 `se2_frozen/` 环境层锁定 **68 维** SE2 观测布局
（`policy_1.pt` 为 68→18 TorchScript，83 维输入无法加载；姊妹任务
`envelope_cascade_83` 锁定的是旧 83 维契约，两者权重**不可**互换）。
`policy_1.pt` 就位后其 md5 会被 `tests/ea2/cascade_68/test_cascade68_contracts.py`
钉死，误替换会在契约测试立即报错。

## MANIFEST

| 文件 | 角色 | 来源 | 校验 |
|---|---|---|---|
| `haa_range.pt` | HAA 范围网络（MLP 8→128→128→12） | `spider_envelop_2/envelop_network/haa_range.pt` 的逐字节拷贝 | md5 `640627fd6a831cfbfcc308b5772101cf`（2026-09-04，与源一致） |
| `ea2_envelope_v3att.pt` | **当前** EA2 感知策略（attitude-replay 重训：真机俯仰/横滚/高度连续回放语料，val R²=0.7375） | `envelope_cascade_83/checkpoints/ea2_envelope_v3att.pt` 的逐字节拷贝（源头 `logs/el4090_ea2/v3_attitude/model_0.pt`） | md5 `d294151ce917180fbbb06cc6aea6c3e1`（2026-09-04，与源一致） |
| `ea2_envelope.pt` | （回滚备用）EA2 感知策略 v2_multik | `envelope_cascade_83/checkpoints/ea2_envelope.pt` 的逐字节拷贝（源头 `logs/el4090_ea2/v2_multik/model_0.pt`） | md5 `4716b023632a5712265fcc672e272306`（2026-09-04，与源一致） |
| `policy_1.pt` | **【占位，尚未就位】** SE2 步态策略（TorchScript，**68**→18，MLP[512,256,128]） | 待 68 维主线 `el4090_envelop_2`（experiment `el_4090_envelop_2_p_haa_range`）训练产物，经 `export_policy_as_jit` 导出后放置于此 | 就位后在此登记 md5，并同步 `tests/ea2/cascade_68/test_cascade68_contracts.py` 的钉死值 |

### policy_1.pt 占位说明（当前空缺）

68 维 SE2 步态权重尚未训练。占位期的行为约定：

- **不放空文件/dummy 文件**——`policy_1.pt` 直接缺位；
- env 构造**不**加载 SE2 策略（仅 `El4090Cascade68Cfg.se2_policy.checkpoint`
  携带指针），`run_cascade68_env_test.py` 与契约/感知测试均无需它；
- `scripts/play_cascade_68.py` 与 `tests/ea2/cascade_68/run_cascade68_closed_loop_test.py`
  在 `torch.jit.load` 前检查文件存在，缺失时报错/以非零码退出并指向本文件；
- 契约测试中的 policy md5 钉死为 `skipif(不存在)`——权重一旦放入本目录即自动
  转为硬钉，无需改测试。

## 加载路径

- `ea2_envelope_v3att.pt`：`El4090Cascade68Cfg.ea2.checkpoint` →
  `ea2_policy.Ea2Policy`（严格 state_dict 加载；遇 `empirical_normalization`
  字段显式报错）。回滚：把 `ea2.checkpoint` 指到 `ea2_envelope.pt`。
- `policy_1.pt`：`El4090Cascade68Cfg.se2_policy.checkpoint` →
  `play_cascade_68.py` / 闭环测试 `torch.jit.load` 直接消费（无 rsl_rl 依赖）。
- `haa_range.pt`：`El4090Cascade68Cfg.haa_swing_range.network_checkpoint`
  **覆盖**了继承的 SE2 指针；`HaaRangeNetwork.from_checkpoint` 会校验
  condition_names 一致性。

## 更换/重训同步流程

1. **EA2 感知**：`sl.scripts.export --run-name <new>` → 拷贝
   `logs/el4090_ea2/<new>/model_0.pt` 到本目录 → 若 soft_dof_pos_limit /
   action_max 有变，同步 `El4090Cascade68Cfg.ea2.fold_scale`（env 构造时断言
   `soft/(2*action_max) == fold_scale`，陈旧折叠会直接抛错）→ 更新本表。
2. **SE2 步态**：68 维主线训练 → `export_policy_as_jit` 导出 → 放置
   `policy_1.pt` → 更新本表 md5 与契约测试钉死值。
3. **HAA 网络**：若 `spider_envelop_2/envelop_network/haa_range.pt` 重训替换，
   重新拷贝到本目录 → 更新本表 md5（与 `envelope_cascade_83/checkpoints/`
   的拷贝两端同步，否则 SE2 基线与两个 cascade 行为分叉）。
