# Package-geometry HAA range distillation

`train_package_haa_range.py` creates a checkpoint compatible with the existing
`HaaRangeNetwork.from_checkpoint()` path.  It retains the deployed tensor
contract: input condition `[B,8]`, output intervals `[B,6,2]`, in HAA order
`RF, RM, RB, LF, LM, LB`.  The labels are instead produced by the capsule
support model in `el4090_envelope`.

The geometric teacher fixes HFE/KFE at the morphology nominal pose, calls
`haa_rejection_ranges`, complements its rejected bands, and chooses one
connected feasible HAA band per leg. The selected band must contain the
morphology preferred HAA. If only a second folded solution is feasible, that
row is rejected instead of switching branch. The selected band is also
intersected with the morphology swing allowance (spider 1.05 rad to mammal
0.45 rad), then shrunken by `--output-margin` so the target does not lie
exactly on a contact boundary. This hard restriction is necessary because one
`[center, half]` pair cannot encode two disjoint HAA solutions safely.

For a quick plumbing test:

```bash
python legged_gym/legged_gym/utils/envelop/train/train_package_haa_range.py \
  --samples 64 --fold-samples 512 --fold-bins 65 --sweep-steps 81 \
  --steps 500 --output /tmp/haa_range_geometry.pt
```

## Bootstrap from the current checkpoint

When no planner-support log exists yet, execute the currently deployed
`spider_envelop_2/envelop_network/haa_range.pt` to create a bootstrap dataset:

```bash
python legged_gym/legged_gym/utils/envelop/train/export_legacy_haa_dataset.py \
  --output /tmp/legacy_haa_bootstrap.pt --samples 10000
```

It saves `condition [N,8]`, the old `legacy_haa_ranges [N,6,2]` for comparison,
and a synthetic symmetric `allowed_support [N,32]`. Feed it into the geometry
trainer:

```bash
python legged_gym/legged_gym/utils/envelop/train/train_package_haa_range.py \
  --support-dataset /tmp/legacy_haa_bootstrap.pt \
  --output /tmp/haa_range_geometry.pt
```

The old ranges are **not** labels for the new student; they are evidence for
comparison only. The synthetic support is useful to validate the pipeline, but
cannot replace logged asymmetric planner/LiDAR supports for final training.

For a usable training checkpoint, generate a representative dataset from the
planner/LiDAR path and prefer:

```bash
python legged_gym/legged_gym/utils/envelop/train/train_package_haa_range.py \
  --support-dataset planner_envelopes.pt \
  --output legged_gym/envs/el_4090/spider_envelop_2/envelop_network/haa_range_geometry.pt
```

The dataset is a Torch dictionary:

```python
{
    "condition": condition,          # float [N,8]; first 5 geometry fields supplied
    "allowed_support": allowed,      # float [N,32]; capsule-envelope half spaces
}
```

The script recomputes morphology priors from the first five condition fields,
so those three supplied columns are ignored and cannot become inconsistent.
It writes checkpoint metadata next to the model as `.pt.json`.

The synthetic mode is only a compatibility bootstrap: it constructs a
symmetric footprint from the legacy 5-D geometry and expands it by
`--synthetic-body-margin`.  It cannot teach the student obstacle asymmetry.
Even with a real asymmetric dataset, an 8-D student cannot distinguish two
different envelopes that have the same 8-D projection.  This is acceptable
only for a drop-in checkpoint; change the runtime observation/network input to
include support features when that information is needed online.

Before configuring `haa_swing_range.network_checkpoint` to use the result,
inspect the saved validation metrics. `unsafe_max_expansion_rad` is the most
important: it measures a predicted lower endpoint below the teacher lower
endpoint, or a predicted upper endpoint above the teacher upper endpoint. A
student range must be made more conservative (larger `--output-margin`, more
data, or an explicit runtime inward shrink) if it is nonzero.
