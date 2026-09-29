# `spider_envelop_3`

`el4090_envelop_3` directly samples six HAA swing intervals at every command
resample. It does not construct a five-parameter body envelope and does not
load an HAA-range checkpoint.

The direct sampler is configured by `El4090Envelop3Cfg.haa_range`. It samples
three intervals, each with endpoints in `[-1.3, 1.3]` and width at least
`0.3 rad`, then copies them to the left/right pairs: `LB/RB`, `LF/RF`, and
`LM/RM`. Each environment stores the resulting values as
`haa_swing_ranges[num_envs, 6, 2]`, in `LB, LF, LM, RB, RF, RM` order.

Training directly samples these intervals without capsule rejection, avoiding
the large GPU allocation required to check every environment at once. PhysX
self-collision is enabled and the active `collision` reward penalizes contacts
on `BASE`, `THIGH`, and `SHANK`, including leg-leg contacts. The standalone
`haa_interference.py` script remains available for offline capsule checks of
sampled boxes; it is a conservative finite-pose check, not a complete
mesh-contact proof for every continuous pose inside an interval.

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

## Observation contract

The policy observation has 79 dimensions:

- 66 standard proprioceptive/command/action values;
- 1 scaled target height (`target_height = 0.52`);
- 12 HAA interval endpoints (six `[lower, upper]` pairs).

The controller, reset posture, joint-position observation reference, and
stand-still reward all use `default_dof_pos + haa_range_bias`. The bias is the
six interval centres and affects HAA only; there is no mammal/default-pose
interpolation in this task.
