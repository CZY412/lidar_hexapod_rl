# checkpoints/ — cascade_83 演示的全部策略权重（pin 进 git，冻结 83 维契约）

演示采取**策略集中管理**：三个权重都收在本目录，运行不依赖 `logs/` 或其他
任务目录。上游重训后请按下表流程同步。

**冻结契约**：本任务与 `se2_frozen/` 环境层锁定 **83 维** SE2 观测布局
（`policy_1.pt` 为 83→18 TorchScript，68 维输入无法加载）。上游 SE2 主线合并
`feat/el_4090_2` 后已转 68 维观测，其新产物**不可**直接替换本表任何权重；
`policy_1.pt` 的 md5 同时被 `tests/ea2/cascade/test_cascade_contracts.py` 钉死，
误替换会在契约测试立即报错。68 维新架构请另建 `el4090_cascade_68` 任务。

## MANIFEST

| 文件 | 角色 | 来源 | 校验 |
|---|---|---|---|
| `ea2_envelope.pt` | （已退役，回滚备用）EA2 感知策略 v2_multik | `legged_gym/logs/el4090_ea2/v2_multik/model_0.pt` | md5 `4716b023632a5712265fcc672e272306` |
| `ea2_envelope_v3att.pt` | **当前** EA2 感知策略（attitude-replay 重训：真机俯仰/横滚/高度连续回放语料，val R²=0.7375） | `legged_gym/logs/el4090_ea2/v3_attitude/model_0.pt` | md5 `d294151ce917180fbbb06cc6aea6c3e1`（2026-09-01） |
| `policy_1.pt` | SE2 步态策略（**TorchScript**，`export_policy_as_jit` 导出；**83**→18，MLP[512,256,128]；仅可被 83 维 obs 消费） | 用户手工放置（训练产物） | md5 `fd2ed3a8caf0ed00cae0ec5d2fcdbea7`（2026-09-01 就位，测试钉死） |
| `haa_range.pt` | HAA 范围网络（MLP 8→128→128→12） | `spider_envelop_2/envelop_network/haa_range.pt` 的逐字节拷贝 | md5 `640627fd6a831cfbfcc308b5772101cf`（与源一致） |

**v3att 语料说明**：EA2 采集启用 `--attitude`（连续体态回放，源
`attitude_traj_source.npz` md5 `39ddb3ab0b7f2ed9ff91d2cb38cfa036`），修复真机
俯仰导致的空地假收缩；v2 语料已归档于 `sl/logs/data_v2_multik/`。

## 加载路径

- `ea2_envelope.pt`：`El4090Cascade83Cfg.ea2.checkpoint` → `ea2_policy.Ea2Policy`
  （严格 state_dict 加载；遇 `empirical_normalization` 字段显式报错）。
- `policy_1.pt`：`El4090Cascade83Cfg.se2_policy.checkpoint` → `play_cascade.py`
  / 测试脚本 `torch.jit.load` 直接消费（无 rsl_rl runner 依赖）。
- `haa_range.pt`：`El4090Cascade83Cfg.haa_swing_range.network_checkpoint`
  **覆盖**了继承的 SE2 指针；加载时 `HaaRangeNetwork.from_checkpoint` 会校验
  condition_names 一致性。

## 更换/重训同步流程

1. **EA2 感知**：`sl.scripts.export --run-name <new>` → 拷贝
   `logs/el4090_ea2/<new>/model_0.pt` 到本目录 → 若 soft_dof_pos_limit /
   action_max 有变，同步 `El4090Cascade83Cfg.ea2.fold_scale`（env 构造时断言
   `soft/(2*action_max) == fold_scale`，陈旧折叠会直接抛错）→ 更新本表。
2. **SE2 步态**：训练侧 `export_policy_as_jit` 导出 → 覆盖 `policy_1.pt`
   → 更新本表（来源与日期）。
3. **HAA 网络**：若 `spider_envelop_2/envelop_network/haa_range.pt` 重训替换，
   重新拷贝到本目录 → 更新本表 md5（两端必须同步，否则 SE2 基线与 cascade
   行为分叉）。
