#!/usr/bin/env python3
"""Interactively compare the HAA range networks used by envelop_2 and 3.

The five sliders are shared geometry inputs.  The three extra inputs required
by envelop_2 are derived from those sliders with the same morphology-prior
rule used by the environment; they are shown above the plot.  Thus the two
models are compared on one consistent envelope condition, rather than on two
unrelated random inputs.

Example (from the inner ``legged_gym`` directory)::

    python utils/envelop/train/visualize_haa_networks_2_vs_3.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[2]
NETWORK_DIR = SCRIPT_DIR.parent / "network"
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))

from haa_swing_range import (  # noqa: E402
    HaaRangeNetwork,
    apply_env_morphology_priors,
    load_envelope_condition_spec,
)
from train_api_haa_range import (  # noqa: E402
    ApiHaaRangeNetwork,
    CONDITION_HIGH,
    CONDITION_LOW,
    CONDITION_NAMES,
)


DISPLAY_LEGS = ("LB", "LF", "LM", "RB", "RF", "RM")
PRIOR_NAMES = (
    "morphology_front_prior",
    "morphology_middle_prior",
    "morphology_back_prior",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-2",
        type=Path,
        default=PACKAGE_ROOT / "envs/el_4090/spider_envelop_2/envelop_network/haa_range.pt",
        help="Eight-input modal/prior HAA network used by envelop_2.",
    )
    parser.add_argument(
        "--checkpoint-3",
        type=Path,
        default=PACKAGE_ROOT / "envs/el_4090/spider_envelop_3/envelop_network/api_haa_range_network.pt",
        help="Five-input API HAA network used by envelop_3.",
    )
    return parser.parse_args()


def load_network_3(path: Path) -> tuple[ApiHaaRangeNetwork, tuple[str, ...]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if tuple(checkpoint["condition_names"]) != CONDITION_NAMES:
        raise ValueError("checkpoint-3 must use envelop_3's five geometry inputs")
    if tuple(checkpoint["output_names"]) != ("lower", "upper"):
        raise ValueError("checkpoint-3 must emit [lower, upper] intervals")
    model = ApiHaaRangeNetwork(tuple(checkpoint["hidden_dims"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, tuple(checkpoint["leg_names"])


def reorder_ranges(
    ranges: torch.Tensor, source_legs: Sequence[str], target_legs: Sequence[str] = DISPLAY_LEGS,
) -> torch.Tensor:
    missing = set(target_legs).difference(source_legs)
    if missing:
        raise ValueError(f"Network output is missing legs: {sorted(missing)}")
    indices = [source_legs.index(leg) for leg in target_legs]
    return ranges[list(indices)]


def main() -> None:
    args = parse_args()
    network_2 = HaaRangeNetwork.from_checkpoint(args.checkpoint_2)
    network_3, legs_3 = load_network_3(args.checkpoint_3)
    legs_2 = tuple(network_2.config.leg_names)
    old_condition_names = tuple(network_2.config.condition_names)
    spec = load_envelope_condition_spec()
    if not set(CONDITION_NAMES).issubset(old_condition_names) or not set(PRIOR_NAMES).issubset(old_condition_names):
        raise ValueError("checkpoint-2 must use five geometry inputs plus three morphology priors")

    initial = 0.5 * (torch.tensor(CONDITION_LOW) + torch.tensor(CONDITION_HIGH))
    fig, axis = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(bottom=0.36, top=0.84)
    colors = {"2": "tab:blue", "3": "tab:orange"}
    prior_text = fig.text(0.12, 0.91, "", fontsize=10)

    def old_condition(geometry: torch.Tensor) -> torch.Tensor:
        """Build envelop_2's eight inputs from the shared five geometry values."""
        values = {name: geometry[index] for index, name in enumerate(CONDITION_NAMES)}
        canonical = torch.stack(
            [values.get(name, torch.tensor(0.0)) for name in spec.condition_names]
        ).unsqueeze(0)
        canonical = apply_env_morphology_priors(canonical, spec)[0]
        canonical_values = {name: canonical[index] for index, name in enumerate(spec.condition_names)}
        return torch.stack([canonical_values[name] for name in old_condition_names])

    @torch.no_grad()
    def predict(geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_2 = old_condition(geometry)
        ranges_2 = reorder_ranges(network_2(condition_2.unsqueeze(0))[0], legs_2)
        ranges_3 = reorder_ranges(network_3(geometry.unsqueeze(0))[0], legs_3)
        return ranges_2, ranges_3, condition_2

    def redraw(_=None) -> None:
        geometry = torch.tensor([slider.val for slider in sliders], dtype=torch.float32)
        ranges_2, ranges_3, condition_2 = predict(geometry)
        axis.clear()
        for index, leg in enumerate(DISPLAY_LEGS):
            low_2, high_2 = ranges_2[index].tolist()
            low_3, high_3 = ranges_3[index].tolist()
            axis.hlines(index + 0.16, low_2, high_2, color=colors["2"], linewidth=7,
                        label="envelop_2 network" if index == 0 else None)
            axis.hlines(index - 0.16, low_3, high_3, color=colors["3"], linewidth=7,
                        label="envelop_3 API network" if index == 0 else None)
            axis.plot((low_2, high_2), (index + 0.16, index + 0.16), "o", color=colors["2"])
            axis.plot((low_3, high_3), (index - 0.16, index - 0.16), "o", color=colors["3"])
            axis.text(3.06, index, f"ΔL={low_3 - low_2:+.3f}, ΔU={high_3 - high_2:+.3f}", va="center", fontsize=8)

        priors = [condition_2[old_condition_names.index(name)].item() for name in PRIOR_NAMES]
        prior_text.set_text(
            "envelop_2 derived priors: " + "   ".join(
                f"{name.replace('morphology_', '').replace('_prior', '')}={value:.3f}"
                for name, value in zip(PRIOR_NAMES, priors)
            )
        )
        axis.axvline(0.0, color="0.4", linestyle="--", linewidth=1)
        axis.set(
            xlim=(-3.05, 4.25), ylim=(-0.7, len(DISPLAY_LEGS) - 0.3),
            yticks=range(len(DISPLAY_LEGS)), yticklabels=DISPLAY_LEGS,
            xlabel="HAA angle [rad]", title="Same geometry: envelop_2 vs envelop_3 HAA range networks",
        )
        axis.grid(axis="x", alpha=0.3)
        axis.legend(loc="upper left")
        fig.canvas.draw_idle()

    sliders = []
    for index, (name, low, high) in enumerate(zip(CONDITION_NAMES, CONDITION_LOW, CONDITION_HIGH)):
        slider_axis = fig.add_axes((0.17, 0.05 + (4 - index) * 0.055, 0.68, 0.025))
        slider = Slider(slider_axis, name, low, high, valinit=float(initial[index]), valstep=0.001)
        slider.on_changed(redraw)
        sliders.append(slider)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
