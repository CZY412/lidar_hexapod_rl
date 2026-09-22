# EL4090 Envelop 3

`el4090_envelop_3` trains the EL4090 inside an allowed body envelope.  It uses
the new `utils/envelop/el4090_envelope` capsule and support-function API; it
does not use the old HAA range network.

The policy has the same 66-dimensional proprioceptive observation as
`envelop_2`.  The envelope geometry, HAA ranges, and morphology priors are
environment-internal state, not extra policy observations.

## Pipeline

At reset and every command resample, the environment executes this pipeline
independently for each simulated robot:

```text
sample five allowed-envelope parameters
        |
        v
construct a symmetric six-vertex polygon in the body XY plane
        |
        v
expand its support function by margin
        |
        v
el4090_envelope API sweeps each leg's HAA spider-to-mammal interval
        |
        v
obtain one safe HAA running interval for each of the six legs
        |
        v
take interval centers and convert them to front/middle/back priors
        |
        v
interpolate spider and mammal HAA/HFE/KFE endpoint poses
        |
        v
use the interpolated pose as embedded_state_default_dof_pos
        |
        v
PD target = embedded_state_default_dof_pos + action * action_scale
```

The policy observation contains, among other proprioception, the relative
joint position:

```text
(dof_pos - embedded_state_default_dof_pos) * obs_scales.dof_pos
```

Therefore the policy sees its joint error relative to the API-derived target,
but it does not receive the five geometry values or the three priors directly.

## Allowed envelope

The external input is the following five parameters from
`El4090Envelop3Cfg.envelope.ranges`:

| Parameter | Range | Meaning |
| --- | --- | --- |
| `front_width` | `[0.3, 0.6]` m | Front half-width: distance from body centerline to the left/right front vertices. |
| `middle_width` | `[0.3, 0.7]` m | Middle half-width. |
| `back_width` | `[0.3, 0.6]` m | Rear half-width. |
| `forward_limit` | `[0.6, 0.9]` m | X coordinate of the two front vertices. |
| `backward_limit` | `[-0.9, -0.6]` m | X coordinate of the two rear vertices. Negative is behind the body origin. |

They create this body-frame polygon, in counter-clockwise order:

```text
(forward_limit,  front_width)
(0,              middle_width)
(backward_limit, back_width)
(backward_limit,-back_width)
(0,             -middle_width)
(forward_limit, -front_width)
```

The new API represents this polygon by its support function.  For each sampled
direction `d`, it computes `max(v dot d)` over the six vertices.  This is the
allowed support before margin.  The robot is represented by collision capsules
from the URDF; its capsule support must fit inside the allowed support in every
tested direction.

Changing these five ranges changes the distribution of allowed envelopes used
for training.  They are the primary morphology/task-design controls.

## API range parameters

`El4090Envelop3Cfg.envelope_api_range` controls numerical evaluation; it does
not alter the nominal six-vertex geometry except for `margin`.

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `directions` | `32` | Number of unit directions used for support-function containment. More directions approximate boundaries more accurately and cost more computation. |
| `margin` | `0.05` m | Safety expansion added to the allowed support in every direction. Positive margin makes the accepted HAA region larger. Set `0.0` for strict containment. |
| `tolerance` | `1e-6` | Floating-point tolerance used by the API's containment/rejection decision. Normally leave unchanged. |
| `sweep_steps` | `201` | Samples per leg when scanning its HAA candidate interval. Higher values give a more accurate boundary but are slower. `51` or `101` are useful for quick debugging. |
| `prior_iterations` | `3` | Fixed-point iterations linking range centers to the pose used by the next API query. More iterations improve consistency but cost extra API calls. |
| `cache_precision` | `3` | Rounds each of the five geometry inputs to `0.001` m for the in-memory result cache. Higher precision has fewer cache hits; lower precision reuses approximate results more often. |
| `sampling_max_attempts` | `32` | Maximum rejection-sampling attempts at command resampling. A sampled polygon is accepted only when all six legs have a safe HAA interval in their modal domain. |

## HAA interval and prior derivation

For every leg, the API searches only the HAA segment between the two configured
endpoint poses.  It never searches the full joint limit merely because it is
physically reachable.

| Leg group | Spider HAA | Mammal HAA |
| --- | ---: | ---: |
| RF, LF | `0.0` | `-1.308` rad |
| RM, LM | `0.0` | `1.308` rad |
| RB, LB | `0.0` | `1.308` rad |

For a selected safe interval `[lo, hi]`, its center is `c = (lo + hi) / 2`.
The leg prior is calculated as:

```text
p_leg = clamp((c - spider_HAA) / (mammal_HAA - spider_HAA), 0, 1)
```

The two legs in each group are averaged:

```text
morphology_front_prior  = mean(p_LF, p_RF)
morphology_middle_prior = mean(p_LM, p_RM)
morphology_back_prior   = mean(p_LB, p_RB)
```

These are derived outputs, never sampled inputs.  Each group prior then
interpolates all three joints of both legs:

```text
target_joint = spider_joint + prior_group * (mammal_joint - spider_joint)
```

This includes HAA, HFE, and KFE.  The endpoint values are in
`El4090Envelop3Cfg.init_state.default_joint_angles` and
`mammal_default_joint_angles`; change them only when deliberately changing the
two modal poses used by the API.

Because the target pose changes the occupied capsules and the capsule shape
changes the HAA range, the calculation is iterative:

1. Start from zero priors (spider endpoint pose).
2. Compute six API HAA ranges inside the chosen allowed envelope.
3. Derive three priors from range centers.
4. Use those priors for the next target pose and API call.
5. Stop after `prior_iterations` iterations.

If an API response has several disjoint safe intervals, the environment first
chooses the interval containing the current modal HAA target.  If none contains
that target, it chooses the widest safe interval.  The HAA range reward uses
this one selected interval per leg.

## Runtime and visualization

Run playback from `legged_gym`:

```bash
python scripts/play_envelop_3.py
```

The playback controls set or sample the five external geometry parameters.
The playback state printout includes the derived three priors and six final HAA
ranges.  With `commands.envelope_debug_viz = True`, the simulator draws:

| Color | Meaning |
| --- | --- |
| Cyan | The five-parameter allowed polygon after `margin` expansion. |
| Orange | The current robot capsule occupied envelope. |

For training, select the registered task `el4090_envelop_3`.  The five geometry
parameters are sampled whenever commands are resampled; `resampling_time` is
inherited from the base envelope task configuration.

Some narrow polygons cannot contain the capsule robot anywhere within the
configured spider-to-mammal HAA intervals.  The API reports this as
`api_mode=none`.  During training, `envelop_3` rejection-samples a replacement
geometry up to `sampling_max_attempts`, so this condition does not terminate a
run.  A manual `set_envelope_condition()` call instead raises a descriptive
error and restores the previous valid geometry; this makes an impossible test
condition visible rather than silently changing it.

## Which settings to change

| Goal | Change |
| --- | --- |
| Train on wider/narrower body space | The three `*_width` ranges. |
| Allow more forward/rear reach | `forward_limit` / `backward_limit` ranges. |
| Add or remove geometric slack | `margin`. |
| Improve range-boundary accuracy | `directions` and/or `sweep_steps`. |
| Improve prior/pose fixed-point consistency | `prior_iterations`. |
| Change the spider or mammal modal posture | The two joint-angle dictionaries, keeping all HAA/HFE/KFE group values intentional. |

Do not treat the three morphology priors as independent configuration or
sampling variables in `envelop_3`: they are determined by the API result.
