# Fixed-Spider HAA Range Network

These scripts train and inspect a neural approximation of the fixed-spider
EL4090 API HAA safe ranges used by `spider_envelop_3`.

The five inputs are `front_width`, `middle_width`, `back_width`,
`forward_limit`, and `backward_limit`. HFE/KFE stay at the v3 spider default
(`0.405`, `-0.39`).
The network output is six safe `[lower, upper]` HAA intervals in the order
`LB, LF, LM, RB, RF, RM`; it is not a rejected-range output.

The label conversion is:

1. Call the capsule/support-function API to get rejected HAA intervals.
2. Take their complement within the configured spider-to-mammal HAA segment.
3. Pin HFE/KFE to the single `spider_envelop_3` default pose (`0.405`,
   `-0.39`), so labels and training use the same posture.
4. Prefer a safe interval containing HAA `0.0`.
5. If zero is unavailable, choose the negative-center interval for front legs
   `LF/RF` (forward splay), or the positive-center interval for middle/rear
   legs `LM/RM/LB/RB` (rearward splay).

## Build the offline feasible-envelope bank

This is required before RL training. It traverses a regular grid over all five
parameter domains, predicts the network range, holds HFE/KFE at v3's default,
and rejects every range whose eight paired-HAA corners collide. `--collision-margin`
inflates the capsule proxy to leave physical clearance.

```bash
python legged_gym/legged_gym/utils/envelop/train/build_feasible_envelope_bank.py \
  --checkpoint legged_gym/legged_gym/envs/el_4090/spider_envelop_3/envelop_network/api_haa_range_network.pt \
  --output legged_gym/legged_gym/envs/el_4090/spider_envelop_3/envelop_network/feasible_envelope_conditions.pt \
  --steps 21 --collision-margin 0.02
```

`--steps 21` means `21^5` points; use `--steps 7` for a quick smoke test, or
pass five values to use different resolutions per parameter.

The output always contains `feasible_mask`: a compact Boolean table for every
grid combination (`False` means that envelope must not be sampled). `conditions`
is the already filtered `True` subset used by RL. Add `--save-rejected` only
when full rejected parameter/range rows are needed for offline analysis.

## Train

From the repository root:

```bash
python legged_gym/legged_gym/utils/envelop/train/train_api_haa_range.py \
  --samples 20000 --epochs 300 \
  --dataset legged_gym/legged_gym/utils/envelop/data/api_haa_ranges.pt \
  --output legged_gym/legged_gym/utils/envelop/data/api_haa_range_network.pt
```

For a quick pipeline check, reduce to `--samples 100 --epochs 5`.

## Visualize

Open the five interactive geometry sliders and inspect network predictions:

```bash
python legged_gym/legged_gym/utils/envelop/train/visualize_api_haa_range_network.py \
  --checkpoint legged_gym/legged_gym/utils/envelop/data/api_haa_range_network.pt
```

Add exact API ranges as thin black overlays for the current slider values:

```bash
python legged_gym/legged_gym/utils/envelop/train/visualize_api_haa_range_network.py \
  --checkpoint legged_gym/legged_gym/utils/envelop/data/api_haa_range_network.pt \
  --compare-api
```

Colored thick bars are network outputs. The vertical dashed line is the spider
default HAA angle `0.0`. The right panel is a URDF-FK top view: the blue
hexagon is the five-parameter envelope, solid legs use each predicted range
centre, and translucent arcs show the available HAA sweep.

`--compare-api` is the validation view: black bars are recomputed directly
from `el4090_envelope`; colored bars are the fitted `haa_range.pt` output.
Add `--check-collision` to show the same all-corner capsule result used by the
training sampler for the current slider setting.
