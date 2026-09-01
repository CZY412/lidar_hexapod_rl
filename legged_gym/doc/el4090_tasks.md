# EL4090 Tasks

## Environment Setup

Training / play commands below assume you are inside:

```bash
cd /home/user/CodeSpace/Diffusion/PredictiveDiffusionPlanner_Dev/extended_legged_gym/legged_gym
conda activate isaacgym
```

Diffusion data-collection commands assume you are inside:

```bash
cd /home/user/CodeSpace/Diffusion/PredictiveDiffusionPlanner_Dev/diffuse_cloc
conda activate isaacgym
```

---

## Task Map

| Behavior | Training Task | Collect Task | `task_vec` | Notes |
| -------- | ------------- | ------------ | ---------- | ----- |
| Normal EL4090 baseline | `el4090_spider_normal` | `el4090_flat_collect` | `[1., 0., 0.]` | Flat collect alias currently points to tripod-2 semantics |
| Tripod 2-gait trot | `el4090_tripod2` | `el4090_tripod2_collect` | `[1., 0., 0.]` | Two synchronized tripod groups |
| Tripod 2-gait trot (low) | `el4090_tripod2_low` | `el4090_tripod2_low_collect` | `[1., 0., 0.]` | Crouching tripod gait, base_height_target=0.22 |
| Tripod 3-gait trot | `el4090_tripod3` | `el4090_tripod3_collect` | `[1., 1., 0.]` | Three alternating foot-pair groups |
| High-standing trot | `el4090_high_stand_trot` | `el4090_high_stand_trot_collect` | `[1., 2., 0.]` | Taller base-height target and posture shaping |
| Wave gait | `el4090_wave` | `el4090_wave_collect` | `[1., 2., 0.]` | Slow stable stepping with one-foot swing emphasis |
| Jump / hop | `el4090_jump` | `el4090_jump_collect` | `[1., 3., 0.]` | Synchronized push-off and aerial phase shaping |
| Mammal gait | `el4090_mammal` | `el4090_mammal_collect` | `[1., 3., 0.]` | Left-right alternating gait approximation |
| Safety task | `el_4090_safe` | - | - | Existing safety-oriented baseline |

---

## Training

### Baseline Evaluation

```bash
python legged_gym/scripts/train.py --task=el4090_spider_normal --num_envs=3072 --headless --resume
python legged_gym/scripts/train.py --task=el_4090_safe --num_envs=3072 --headless --resume
```

### Multi-Behavior EL4090 Training

Use `3072` to `4096` envs as the default starting range. For first-pass debugging, use `--num_envs=512` or lower.

```bash
python legged_gym/scripts/train.py --task=el4090_tripod2 --num_envs=4096 --headless --resume
python legged_gym/scripts/train.py --task=el4090_tripod2_low --num_envs=4096 --headless --resume
python legged_gym/scripts/train.py --task=el4090_tripod3 --num_envs=4096 --headless --resume
python legged_gym/scripts/train.py --task=el4090_high_stand_trot --num_envs=4096 --headless --resume
python legged_gym/scripts/train.py --task=el4090_wave --num_envs=4096 --headless --resume
python legged_gym/scripts/train.py --task=el4090_jump --num_envs=4096 --headless --resume
python legged_gym/scripts/train.py --task=el4090_mammal --num_envs=4096 --headless --resume
```

### Smoke-Test Training

Use this before long runs to validate task wiring:

```bash
python legged_gym/scripts/train.py --task=el4090_tripod2 --num_envs=64 --headless --max_iterations=1
python legged_gym/scripts/train.py --task=el4090_tripod2_low --num_envs=64 --headless --max_iterations=1
python legged_gym/scripts/train.py --task=el4090_wave --num_envs=64 --headless --max_iterations=1
python legged_gym/scripts/train.py --task=el4090_jump --num_envs=64 --headless --max_iterations=1
```

---

## Evaluation / Play

The usual evaluation loop is `play.py` with `--checkpoint=-1` and either a specific `--load_run` or the latest run.

### Baselines

```bash
python legged_gym/scripts/play.py --task=el4090_spider_normal --num_envs=48 --checkpoint=-1 --resume
python legged_gym/scripts/play.py --task=el_4090_safe --num_envs=48 --checkpoint=-1 --resume
```

### 使用独显模式
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json python legged_gym/legged_gym/scripts/play.py --task=el4090_tripod2_low --num_envs=12 --checkpoint=-1
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json python legged_gym/legged_gym/scripts/play.py --task=el4090_lidar_tripod2_low --num_envs 16 --checkpoint -1 --load_run 4090-1
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json python legged_gym/legged_gym/scripts/play_ea2.py --task el4090_ea2 --num_envs 1
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json && python legged_gym/legged_gym/scripts/play_ea2.py --task=el4090_ea2 --load_run baseline --checkpoint 0 --num_envs 1
```

### Behavior Policies

```bash
python legged_gym/scripts/play.py --task=el4090_tripod2 --num_envs=48 --checkpoint=-1 --resume
python legged_gym/scripts/play.py --task=el4090_tripod2_low --num_envs=48 --checkpoint=-1 --resume
python legged_gym/scripts/play.py --task=el4090_tripod3 --num_envs=48 --checkpoint=-1 --resume
python legged_gym/scripts/play.py --task=el4090_high_stand_trot --num_envs=48 --checkpoint=-1 --resume
python legged_gym/scripts/play.py --task=el4090_wave --num_envs=48 --checkpoint=-1 --resume
python legged_gym/scripts/play.py --task=el4090_jump --num_envs=48 --checkpoint=-1 --resume
python legged_gym/scripts/play.py --task=el4090_mammal --num_envs=48 --checkpoint=-1 --resume
```

### Evaluate A Specific Run

```bash
python legged_gym/scripts/play.py --task=el4090_tripod2 --num_envs=48 --checkpoint=-1 --load_run=May07_12-00-00_ --resume
python legged_gym/scripts/play.py --task=el4090_wave --num_envs=48 --checkpoint=-1 --load_run=May07_13-00-00_ --resume
```

---

## Diffusion Data Collection

All EL4090 collect tasks are compatible with:

```bash
python scripts/data_collection_gym/collect.py --tasks <collect_task> --checkpoints <checkpoint_path> --output <zarr_path> --num_envs 256 --len_to_save 500000
```

### Common Examples

```bash
python scripts/data_collection_gym/collect.py \
    --tasks el4090_tripod2_collect \
    --checkpoints /path/to/el4090_tripod2.pt \
    --output data/legged_gym/el4090_tripod2.zarr \
    --num_envs 256 --len_to_save 500000

python scripts/data_collection_gym/collect.py \
    --tasks el4090_tripod2_low_collect \
    --checkpoints /path/to/el4090_tripod2_low.pt \
    --output data/legged_gym/el4090_tripod2_low.zarr \
    --num_envs 256 --len_to_save 500000

python scripts/data_collection_gym/collect.py \
    --tasks el4090_tripod3_collect \
    --checkpoints /path/to/el4090_tripod3.pt \
    --output data/legged_gym/el4090_tripod3.zarr \
    --num_envs 256 --len_to_save 500000

python scripts/data_collection_gym/collect.py \
    --tasks el4090_wave_collect \
    --checkpoints /path/to/el4090_wave.pt \
    --output data/legged_gym/el4090_wave.zarr \
    --num_envs 256 --len_to_save 500000
```

### Multi-Task Collection

```bash
python scripts/data_collection_gym/collect.py \
    --tasks el4090_tripod2_collect \
    --tasks el4090_tripod3_collect \
    --tasks el4090_wave_collect \
    --checkpoints /path/to/el4090_tripod2.pt \
    --checkpoints /path/to/el4090_tripod3.pt \
    --checkpoints /path/to/el4090_wave.pt \
    --output data/legged_gym/el4090_multi_behavior.zarr \
    --num_envs 256 --len_to_save 900000
```

### Current Alias

`el4090_flat_collect` is kept as a backward-compatible alias and currently follows the tripod-2 collect setup.

```bash
python scripts/data_collection_gym/collect.py \
    --tasks el4090_flat_collect \
    --checkpoints /path/to/el4090_tripod2.pt \
    --output data/legged_gym/el4090_flat_alias.zarr \
    --num_envs 256 --len_to_save 500000
```

---

## Recommended Workflow

1. Run a `--max_iterations=1` smoke test for the task.
2. Train with `4096` envs headless.
3. Inspect the latest checkpoint with `play.py` using `--num_envs=48`.
4. Once gait quality is acceptable, collect a short `10k` to `50k` zarr first.
5. Only then start long data-collection jobs.

---

## Envelope Tasks Landscape (post feat/el_4090_2 merge, 2026-09)

| Task | SE2 侧观测 | 状态 |
|---|---|---|
| `el4090_envelop_2`（SE2 主线） | **68 维** | 合并后主线：range priors 已从策略观测移除（主分支 `6cb4e49`）。旧 83 维 checkpoint 不可加载；重训出 68 维权重前 `play_envelop_2.py` 无可用 checkpoint |
| `el4090_ea2` | —（190 维感知） | 活跃主线：SL 数据采集与训练不受合并影响 |
| `el4090_cascade_83` | **83 维（冻结）** | 遗留合并演示：EA2 感知 → 冻结 SE2 步态（`se2_frozen/` + `policy_1.pt` TorchScript，md5 钉死于契约测试）。`play_cascade.py --task el4090_cascade_83` |
| `el4090_cascade_68`（规划中） | 68 维 | 未来基于新 SE2 架构 + 新训练 policy 的级联任务；EA2 GRU 权重与 `EnvelopeBridge` 可直接复用（`set_envelope_condition` 接口在 68 维主线保留） |

- envelope 数学库已包化为 `el4090_envelope/`：跑包内测试前 `pip install -e ./el4090_envelope`；旧路径 `legged_gym.utils.envelop.kinematic_envelope` 仍为兼容 facade（注意本仓库的既有导入顺序约定：先 import `legged_gym.envs` 下的模块，再 import `legged_gym.utils.*`）。
- `el4090_cascade_83` 的 PPO 训练目录独立为 `el4090_cascade_83_p_haa_range`，与 SE2 主线目录 `el_4090_envelop_2_p_haa_range` 隔离，防止 68 维产物被 83 维任务误加载。
- 细节见 `envs/el_4090/envelope_cascade_83/README.md` 及其 `checkpoints/README.md`。

---

## Notes

- `wave`, `jump`, and `mammal` are first-pass behavior tasks. Expect at least one reward-scale tuning pass after visual evaluation.
- If a behavior looks unstable in `play.py`, reduce command ranges first before changing network size.
- For diffusion collection, prefer dedicated collect task names over the flat alias so the dataset metadata stays explicit.
