# Fixed-Spider HAA Range Network

These scripts train and inspect a neural approximation of the fixed-spider
EL4090 API HAA safe ranges used by `spider_envelop_3`.

The five inputs are `front_width`, `middle_width`, `back_width`,
`forward_limit`, and `backward_limit`. HFE/KFE stay at spider default zero.
The network output is six safe `[lower, upper]` HAA intervals in the order
`LB, LF, LM, RB, RF, RM`; it is not a rejected-range output.

The label conversion is:

1. Call the capsule/support-function API to get rejected HAA intervals.
2. Take their complement within the configured spider-to-mammal HAA segment.
3. Prefer a safe interval containing HAA `0.0`.
4. If zero is unavailable, choose the negative-center interval for front legs
   `LF/RF` (forward splay), or the positive-center interval for middle/rear
   legs `LM/RM/LB/RB` (rearward splay).

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
default HAA angle `0.0`.
