#!/usr/bin/env python3
"""Render pairwise 2-D feasibility maps from the offline envelope bank.

The collision decision is five-dimensional, so a two-dimensional plot cannot
honestly label a point simply safe/unsafe without fixing three values.  Each
panel therefore shows P(feasible | displayed two parameters): the fraction of
all grid points along the other three parameter axes that survived collision.
Dark green means a robustly feasible region; white/red means that nearly every
combination of the hidden three parameters was rejected.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="PNG output; omit to open an interactive window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = torch.load(args.bank, map_location="cpu")
    names = tuple(bank["condition_names"])
    axes = tuple(torch.as_tensor(axis).numpy() for axis in bank["grid_axes"])
    shape = tuple(int(value) for value in bank["grid_steps"])
    mask = torch.as_tensor(bank["feasible_mask"], dtype=torch.float32)
    if len(names) != 5 or mask.numel() != int(np.prod(shape)):
        raise ValueError("bank must contain a complete five-dimensional feasible_mask")
    mask = mask.reshape(shape).numpy()
    total, valid = mask.size, int(mask.sum())
    pair_probabilities = []
    for x_index, y_index in combinations(range(5), 2):
        hidden = tuple(index for index in range(5) if index not in (x_index, y_index))
        probability = mask.mean(axis=hidden)
        if [index for index in range(5) if index not in hidden] != [x_index, y_index]:
            probability = probability.T
        pair_probabilities.append((x_index, y_index, probability))
    # A global 0--100% color scale hides every useful signal here because this
    # bank is deliberately restrictive.  The color bar still reports the real
    # fraction, but stretches its upper bound to the best observed 2-D cell.
    maximum = max(float(probability.max()) for _, _, probability in pair_probabilities)
    fig, panels = plt.subplots(2, 5, figsize=(22, 8), constrained_layout=True)
    panels = panels.ravel()
    image = None
    for panel, (x_index, y_index, probability) in zip(panels, pair_probabilities):
        image = panel.imshow(
            probability, origin="lower", aspect="auto", vmin=0., vmax=maximum, cmap="RdYlGn",
            extent=(axes[x_index][0], axes[x_index][-1], axes[y_index][0], axes[y_index][-1]),
        )
        panel.set(title=f"feasible fraction: {names[x_index]} × {names[y_index]}",
                  xlabel=names[x_index], ylabel=names[y_index])
        panel.contour(probability, levels=[.01, .05, .10], colors="black", linewidths=.7,
                      origin="lower", extent=(axes[x_index][0], axes[x_index][-1], axes[y_index][0], axes[y_index][-1]))
    colorbar = fig.colorbar(image, ax=panels.tolist(), shrink=.8, pad=.01)
    colorbar.set_label("actual feasible fraction over remaining three parameters")
    fig.suptitle(f"Five-parameter envelope feasibility: {valid:,}/{total:,} grid points ({valid / total:.3%})", fontsize=16)
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved pairwise feasibility map to {args.output}")


if __name__ == "__main__":
    main()
