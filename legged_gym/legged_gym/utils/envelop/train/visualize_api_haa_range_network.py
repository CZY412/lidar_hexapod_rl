#!/usr/bin/env python3
"""Interactively inspect fixed-spider API HAA range-network predictions.

Example from the repository root:

    python legged_gym/utils/envelop/train/visualize_api_haa_range_network.py \
        --checkpoint legged_gym/utils/envelop/data/api_haa_range_network.pt

Move the five sliders to change the allowed polygon. Colored horizontal bars
update immediately: their endpoints are the network's safe HAA range outputs.
Use ``--compare-api`` to overlay the exact API labels in black.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch

from train_api_haa_range import (
    ApiHaaRangeNetwork,
    ApiRangeLabeler,
    CONDITION_HIGH,
    CONDITION_LOW,
    CONDITION_NAMES,
    LEG_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compare-api", action="store_true")
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--sweep-steps", type=int, default=201)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if tuple(checkpoint.get("condition_names", ())) != CONDITION_NAMES:
        raise ValueError("Checkpoint does not use envelop_3's five geometry inputs")
    if tuple(checkpoint.get("output_names", ())) != ("lower", "upper"):
        raise ValueError("Checkpoint uses the old modal/prior network format; retrain with train_api_haa_range.py")
    model = ApiHaaRangeNetwork(tuple(checkpoint["hidden_dims"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    labeler = None
    if args.compare_api:
        labeler = ApiRangeLabeler(
            directions=args.directions, margin=args.margin, tolerance=1e-6, sweep_steps=args.sweep_steps,
        )

    initial = 0.5 * (torch.tensor(CONDITION_LOW) + torch.tensor(CONDITION_HIGH))
    fig, axis = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.36)
    colors = plt.cm.tab10.colors

    def predict(values):
        with torch.no_grad():
            return model(torch.tensor(values, dtype=torch.float32).unsqueeze(0))[0].numpy()

    def redraw(_=None):
        values = [slider.val for slider in sliders]
        prediction = predict(values)
        axis.clear()
        for index, leg in enumerate(LEG_NAMES):
            low, high = prediction[index]
            axis.hlines(index, low, high, color=colors[index], linewidth=9, label="network" if index == 0 else None)
            axis.plot((low, high), (index, index), "o", color=colors[index])
        if labeler is not None:
            try:
                truth = labeler.label(torch.tensor(values)).numpy()
                for index in range(len(LEG_NAMES)):
                    axis.hlines(index, truth[index, 0], truth[index, 1], color="black", linewidth=2,
                                label="API truth" if index == 0 else None)
            except labeler.no_feasible as error:
                axis.text(0.02, 0.03, f"API has no feasible HAA posture: {error}", transform=axis.transAxes, color="crimson")
        axis.axvline(0.0, color="0.4", linestyle="--", linewidth=1, label="spider default HAA")
        axis.set(xlim=(-3.05, 3.05), ylim=(-0.7, len(LEG_NAMES) - 0.3), yticks=range(len(LEG_NAMES)), yticklabels=LEG_NAMES,
                 xlabel="HAA angle [rad]", title="Fixed-spider HAA safe ranges")
        axis.grid(axis="x", alpha=0.3)
        axis.legend(loc="upper right")
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
