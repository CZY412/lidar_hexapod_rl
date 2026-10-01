# `spider_envelop_3`

`el4090_envelop_3` samples a five-parameter body envelope at every episode
reset, evaluates the configured HAA range checkpoint, and uses its six
predicted HAA swing intervals. It therefore no longer samples unrelated HAA
endpoints directly.

The sampler freely draws five parameters from the checkpoint's configured input
domain. It retries only if a predicted HAA interval is non-finite or narrower
than `0.3 rad`; it does no collision geometry work during RL. Left/right pairs
`LB/RB`, `LF/RF`, and `LM/RM` share one interval. Each environment stores
resulting values as
`haa_swing_ranges[num_envs, 6, 2]`, in `LB, LF, LM, RB, RF, RM` order.

The optional offline checker evaluates only the default/bias pose (the centre
of each paired HAA interval), with an extra capsule-radius margin. PhysX
self-collision is also enabled during rollouts; the capsule result is a
conservative finite-pose filter, not a complete mesh-contact proof.

## Setup and training commands

Run these commands from the repository root. The checkpoint path below is the
default in `El4090Envelop3Cfg.haa_range`.

```bash
# 1. Generate exact API labels and train the 5-input -> 6x[lower, upper] net.
python legged_gym/legged_gym/utils/envelop/train/train_api_haa_range.py \
  --samples 20000 --epochs 300 \
  --dataset legged_gym/legged_gym/utils/envelop/data/api_haa_ranges.pt \
  --output legged_gym/legged_gym/envs/el_4090/spider_envelop_3/envelop_network/api_haa_range_network.pt

# 2. Optional offline diagnosis only: map collision-free/rejected parameter
#    combinations. This file is not loaded by the RL environment.
python legged_gym/legged_gym/utils/envelop/train/build_feasible_envelope_bank.py \
  --checkpoint legged_gym/legged_gym/envs/el_4090/spider_envelop_3/envelop_network/api_haa_range_network.pt \
  --output legged_gym/legged_gym/envs/el_4090/spider_envelop_3/envelop_network/feasible_envelope_conditions.pt \
  --steps 21 --collision-margin 0.02

# 3. Inspect the fitted net against the exact API and the robot top view.
python legged_gym/legged_gym/utils/envelop/train/visualize_api_haa_range_network.py \
  --checkpoint legged_gym/legged_gym/envs/el_4090/spider_envelop_3/envelop_network/api_haa_range_network.pt \
  --compare-api --check-collision

# 4. Run RL. It samples five parameters freely; there is no collision check.
cd legged_gym/legged_gym
python scripts/train.py --task el4090_envelop_3

# 5. Interactive playback. R performs the same free five-parameter sampling
#    and range-network inference as training; collision is not checked.
python scripts/play_envelop_3.py
```

`--steps 21` is a `21^5` grid. Start with `--steps 7` for a quick diagnostic
check. If you use the offline collision report after changing the checkpoint,
rebuild it so the report describes the new model.

## Runtime data flow

At initialization, `EnvelopeRangeState` loads `api_haa_range_network.pt`. At
every episode reset it uniformly samples five inputs within that checkpoint's
input bounds, passes them through the network, and stores the result in
`haa_swing_ranges[num_envs, 6, 2]`. It retries only until all predicted ranges
have width at least `0.3 rad`. It also stores the selected input as
`env.envelope_parameters[num_envs, 5]`. The policy observes the 12 range
endpoints, not the five raw envelope values. Collision is neither imported nor
evaluated in this rollout path.

The optional offline collision report has `feasible_mask` and `grid_axes` for
analysis, but it does not constrain runtime sampling.

## HAA range reward

The range is a constraint, not an HAA trajectory reference. A small smooth
compliance reward applies anywhere in the interval interior. A Huber penalty
applies outside it and ramps from 10% to its configured strength over the first
100,000 control steps. There is no mandatory HAA swing reward, so locomotion
and foot-contact rewards may choose the gait that works best inside the range.

The interval centre supplies a separate **PD posture bias**. It is added only
to the six HAA entries of the immutable spider `default_dof_pos`; HFE/KFE keep
their spider defaults. Therefore zero action, reset posture, joint-position
observation error, and standing reward are centered on the sampled range,
without overwriting the static default posture or restoring morphology priors.

## `default_dof_pos` and bias implementation

`default_dof_pos` is created once from `init_state.default_joint_angles` and is
never written by v3. Let the predicted interval for leg `i` be `[l_i, u_i]`.
The only new value is `haa_range_bias[i] = center_bias_scale * (l_i + u_i)/2`.
It is written into the corresponding six HAA columns only; all HFE/KFE entries
remain zero in the bias. The actual target used by v3 is always:

```text
condition_target = default_dof_pos + haa_range_bias
```

It is used in four places:

1. PD torque target: `action * action_scale + condition_target`.
2. Reset: selected environments are placed at `condition_target` with zero
   velocity after their envelope/range is sampled.
3. Proprioceptive joint-position error: `dof_pos - condition_target`.
4. Standing-pose reward: distance from `condition_target`.

`embedded_state_default_dof_pos` is a per-environment cache of the same target
for parent-task compatibility and viewer/reset bookkeeping. It is refreshed on
initialization, explicit range replacement, range sampling, and reset. It is
not the source of the PD target; v3 recomputes `default_dof_pos + bias`, so the
immutable static default cannot be accidentally overwritten.

## Observation contract

The policy observation has 79 dimensions:

- 66 standard proprioceptive/command/action values;
- 1 scaled target height (`target_height = 0.55`);
- 12 HAA interval endpoints (six `[lower, upper]` pairs).

The controller, reset posture, joint-position observation reference, and
stand-still reward all use `default_dof_pos + haa_range_bias`. The bias is the
six interval centres and affects HAA only; there is no mammal/default-pose
interpolation in this task.
