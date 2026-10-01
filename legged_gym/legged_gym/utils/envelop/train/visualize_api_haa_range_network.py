#!/usr/bin/env python3
"""Interactively inspect fixed-spider API HAA range-network predictions.

Example from the repository root:

    python legged_gym/utils/envelop/train/visualize_api_haa_range_network.py \
        --checkpoint legged_gym/utils/envelop/data/api_haa_range_network.pt

Move the five sliders to change the allowed polygon. Colored horizontal bars
and the URDF-FK top-view robot update immediately: transparent arcs are the
predicted HAA motion ranges and each solid leg is posed at its range centre.
Use ``--compare-api`` to overlay the exact API labels in black.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    parser.add_argument("--check-collision", action="store_true",
                        help="deprecated: collision diagnosis is enabled by default")
    parser.add_argument("--no-collision-diagnostic", action="store_true",
                        help="disable the offline capsule diagnostic in the interactive view")
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
    collision_checker = None
    if not args.no_collision_diagnostic:
        env_dir = Path(__file__).resolve().parents[3] / "envs/el_4090/spider_envelop_3"
        if str(env_dir) not in sys.path:
            sys.path.insert(0, str(env_dir))
        from haa_interference import El4090HaaInterferenceChecker
        dof_names = [f"{leg}_{joint}" for leg in LEG_NAMES for joint in ("HAA", "HFE", "KFE")]
        default = torch.tensor([0.405 if name.endswith("HFE") else -0.39 if name.endswith("KFE") else 0.
                                for name in dof_names])
        collision_checker = El4090HaaInterferenceChecker(dof_names, default)

    # Use the same URDF FK as the envelope algorithm.  The right panel is not
    # a cartoon: its six skeletons and HAA arcs come from the actual model.
    source = Path(__file__).resolve().parents[1] / "el4090_envelope" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from el4090_envelope import BatchedUrdfKinematics, load_urdf_joints
    from el4090_envelope.geometry import haa_arc_geometry
    urdf = Path(__file__).resolve().parents[4] / "resources/robots/el_4090/urdf/el_4090.urdf"
    kinematics = BatchedUrdfKinematics(load_urdf_joints(urdf))
    joint_names = kinematics.actuated_joint_names
    haa_indices = [joint_names.index(f"{leg}_HAA") for leg in LEG_NAMES]
    default_q = torch.zeros(kinematics.num_dof)
    for leg in LEG_NAMES:
        default_q[joint_names.index(f"{leg}_HFE")] = .405
        default_q[joint_names.index(f"{leg}_KFE")] = -.39

    initial = 0.5 * (torch.tensor(CONDITION_LOW) + torch.tensor(CONDITION_HIGH))
    fig, (axis, top_axis) = plt.subplots(1, 2, figsize=(15, 7))
    fig.subplots_adjust(bottom=0.36)
    colors = plt.cm.tab10.colors

    def predict(values):
        with torch.no_grad():
            return model(torch.tensor(values, dtype=torch.float32).unsqueeze(0))[0].numpy()

    def draw_top_view(values, prediction, runtime_ok, collision_ok):
        """Top view: allowed envelope, exact FK skeleton, and HAA sweep arcs."""
        top_axis.clear()
        front, middle, back, forward, backward = values
        envelope = np.array(((forward, front), (0., middle), (backward, back),
                             (backward, -back), (0., -middle), (forward, -front),
                             (forward, front)))
        top_axis.fill(envelope[:, 0], envelope[:, 1], color="#8ecae6", alpha=.20,
                      label="5-parameter envelope")
        top_axis.plot(envelope[:, 0], envelope[:, 1], "--", color="#1677a5", linewidth=2)
        # The robot is posed at the predicted interval centre.  The transparent
        # arcs show every HAA angle available to the policy for that leg.
        ranges = torch.tensor(prediction, dtype=torch.float32)
        q = default_q.clone()
        q[haa_indices] = ranges.mean(dim=1)
        _, arcs, markers = haa_arc_geometry(kinematics, q, ranges, radius=.22, samples=41)
        for index, leg in enumerate(LEG_NAMES):
            links = (f"{leg}_HIP", f"{leg}_THIGH", f"{leg}_SHANK", f"{leg}_FOOT")
            local = torch.zeros((len(links), 1, 3))
            points = kinematics.points(q.unsqueeze(0), links, local)[0, :, 0, :2].numpy()
            color = colors[index]
            top_axis.plot(points[:, 0], points[:, 1], "-o", color=color, linewidth=3,
                          markersize=4, label=f"{leg} leg" if index == 0 else None)
            arc = arcs[index, :, :2].numpy()
            top_axis.plot(arc[:, 0], arc[:, 1], color=color, linewidth=6, alpha=.35)
            top_axis.plot(markers[index, 0].item(), markers[index, 1].item(), "o", color=color,
                          markeredgecolor="black", markersize=6)
            top_axis.text(points[-1, 0], points[-1, 1],
                          f"{leg}\n[{ranges[index, 0]:.2f}, {ranges[index, 1]:.2f}]",
                          color=color, fontsize=8, ha="center", va="bottom")
        top_axis.add_patch(plt.Rectangle((-.20, -.12), .40, .24, fill=True,
                                         facecolor="#263238", edgecolor="black", alpha=.8))
        top_axis.plot(0., 0., "+", color="white", markersize=9, markeredgewidth=2)
        diagnostic = "N/A" if collision_ok is None else ("PASS" if collision_ok else "REJECT")
        top_axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal",
                     xlabel="body x [m]", ylabel="body y [m]",
                     title=f"Top view | v3 range: {'ACCEPT' if runtime_ok else 'RETRY'} | collision: {diagnostic}")
        top_axis.title.set_color("forestgreen" if runtime_ok and collision_ok is not False else "crimson")
        top_axis.grid(alpha=.25)
        top_axis.legend(loc="upper right", fontsize=8)

    def redraw(_=None):
        values = [slider.val for slider in sliders]
        prediction = predict(values)
        widths = prediction[:, 1] - prediction[:, 0]
        runtime_ok = bool(np.isfinite(prediction).all() and np.all(widths >= .3 - 1e-6))
        collision_ok = None if collision_checker is None else bool(
            collision_checker.is_safe(torch.tensor(prediction).unsqueeze(0)).item()
        )
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
        runtime_text = "v3 free sampling: ACCEPT" if runtime_ok else "v3 free sampling: RETRY (range width < 0.3 rad)"
        axis.text(0.02, 0.09, runtime_text, transform=axis.transAxes,
                  color="forestgreen" if runtime_ok else "crimson", fontweight="bold")
        if collision_ok is not None:
            axis.text(0.02, 0.03, "offline collision diagnostic: PASS" if collision_ok else "offline collision diagnostic: REJECT",
                      transform=axis.transAxes, color="forestgreen" if collision_ok else "crimson", fontweight="bold")
        axis.axvline(0.0, color="0.4", linestyle="--", linewidth=1, label="spider default HAA")
        axis.set(xlim=(-3.05, 3.05), ylim=(-0.7, len(LEG_NAMES) - 0.3), yticks=range(len(LEG_NAMES)), yticklabels=LEG_NAMES,
                 xlabel="HAA angle [rad]", title="Fixed-spider HAA safe ranges")
        axis.grid(axis="x", alpha=0.3)
        axis.legend(loc="upper right")
        draw_top_view(values, prediction, runtime_ok, collision_ok)
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
