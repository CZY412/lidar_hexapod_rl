#!/usr/bin/env python
"""Top-view snapshots comparing the policy envelope against the oracle target.

Companion to ``validate_sl_vs_oracle_audit.py``: picks representative frames
from the collected zero-action rollouts and renders the occupancy grid with

  * policy envelope      (cyan, solid)
  * smoothed oracle target (orange, solid)
  * minimum envelope     (grey, dashed)

for three frame categories: network-only collisions (target safe, policy
unsafe), floor-pinned geometry, and deep-shrink under-attenuation.

Run:  python legged_gym/tests/ea2/validate_sl_vs_oracle_snapshots.py
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import torch

try:
    from . import _ea2_testlib as tl
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _ea2_testlib as tl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    compute_hex_vertices,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import data_path

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_outputs")


def draw_frame(ax, occ, df, pos_xy, heading, hexes, title, half=4.0):
    cx, cy = float(pos_xy[0]), float(pos_xy[1])
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    ix0 = max(0, int((x0 - tl.WORLD_MIN) / tl.RES))
    ix1 = min(occ.shape[1], int((x1 - tl.WORLD_MIN) / tl.RES) + 1)
    iy0 = max(0, int((y0 - tl.WORLD_MIN) / tl.RES))
    iy1 = min(occ.shape[0], int((y1 - tl.WORLD_MIN) / tl.RES) + 1)
    crop = occ[iy0:iy1, ix0:ix1]
    ax.imshow(
        crop, origin="lower", cmap="Greys", alpha=0.55, vmin=0, vmax=1,
        extent=[tl.WORLD_MIN + ix0 * tl.RES, tl.WORLD_MIN + ix1 * tl.RES,
                tl.WORLD_MIN + iy0 * tl.RES, tl.WORLD_MIN + iy1 * tl.RES],
    )
    # clearance contour at the collision margin, from the distance field
    ax.contour(
        df[iy0:iy1, ix0:ix1], levels=[0.10], origin="lower",
        extent=[tl.WORLD_MIN + ix0 * tl.RES, tl.WORLD_MIN + ix1 * tl.RES,
                tl.WORLD_MIN + iy0 * tl.RES, tl.WORLD_MIN + iy1 * tl.RES],
        colors="gray", linewidths=0.6, alpha=0.8,
    )
    styles = [
        ("policy", "#00b8d4", "-", 2.0),
        ("oracle target", "#ff8c00", "-", 2.0),
        ("min envelope", "#888888", "--", 1.2),
        ("max envelope (config)", "#2e7d32", ":", 1.2),
    ]
    for (name, verts), (label, color, ls, lw) in zip(hexes, styles):
        ax.add_patch(Polygon(verts, closed=True, fill=False,
                             edgecolor=color, linestyle=ls, linewidth=lw, label=label))
    ax.plot([cx], [cy], "k+", markersize=8)
    ax.arrow(cx, cy, 0.6 * float(torch.cos(heading)), 0.6 * float(torch.sin(heading)),
             head_width=0.15, head_length=0.25, fc="k", ec="k")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=7)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../envs/el_4090/envelope_adaptive_2/sl/logs/runs/baseline/model.pt",
    ))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-per-cat", type=int, default=2)
    args = ap.parse_args(argv)

    audit = importlib.import_module("validate_sl_vs_oracle_audit")

    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.dataset import load_map
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import load_checkpoint

    net, _ = load_checkpoint(args.ckpt, device=args.device)
    device = args.device
    data = load_map(data_path(args.seed))
    df = data.distance_field.to(device)
    occ = (df <= 1e-4).cpu().numpy()  # obstacle cells
    head = data.heading.to(device)
    pos = data.pos.to(device)
    s_tgt = data.target.to(device)

    t_len, n_envs = head.shape
    h_f, p_f = head.reshape(-1), pos.reshape(-1, 2)
    s_tgt_f = s_tgt.reshape(-1, 5)
    s_raw = ((audit.raw_oracle(h_f, p_f, df) - audit.MIN_V.to(device))
             / audit.SPAN.to(device)).clamp(0.0, 1.0)
    s_pol = audit.stateful_predict(net, data.obs, device).reshape(-1, 5)
    c_tgt = audit.min_clearance(audit.s_to_params(s_tgt_f), h_f, p_f, df)
    c_pol = audit.min_clearance(audit.s_to_params(s_pol), h_f, p_f, df)
    c_floor = audit.min_clearance(
        audit.MIN_V.to(device).expand(s_pol.shape[0], 5), h_f, p_f, df
    )

    valid = torch.zeros(t_len, n_envs, dtype=torch.bool, device=device)
    meta = data.meta
    warmup = max(1, int(meta.get("warmup_steps", 30)) // max(1, int(meta.get("lidar_decimation", 1))))
    valid[warmup:] = True
    valid = valid.reshape(-1)

    def hexes_for(i: int):
        out = []
        for s in (s_pol[i], s_tgt_f[i], audit.MIN_V.to(device), torch.ones(5, device=device)):
            p = audit.s_to_params(s.unsqueeze(0))[0]
            # scalar inputs -> compute_hex_vertices returns (6, 2) directly
            verts = compute_hex_vertices(p[0], p[1], p[2], p[3], p[4])
            cos_h, sin_h = float(torch.cos(h_f[i])), float(torch.sin(h_f[i]))
            rot = torch.tensor([[cos_h, -sin_h], [sin_h, cos_h]], device=verts.device)
            world = verts @ rot.T + p_f[i]
            out.append((None, world.cpu().numpy()))
        return out

    smin_tgt = s_tgt_f.min(dim=-1).values
    cats = {
        "network_only_collision": valid & (c_pol < 0.10) & (c_tgt >= 0.10),
        "floor_pinned": valid & (c_floor < 0.095),
        "deep_shrink_underatten": valid & (smin_tgt < 0.2) & (c_pol >= 0.10),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    for cat, mask in cats.items():
        idxs = mask.nonzero(as_tuple=False).flatten()
        if idxs.numel() == 0:
            print(f"[snap] {cat}: no frames")
            continue
        # rank by severity: deepest policy violation first (or largest gap)
        if cat == "deep_shrink_underatten":
            score = (s_tgt_f[:, :] - s_pol).min(dim=-1).values
        else:
            score = -c_pol
        order = idxs[torch.argsort(score[idxs], descending=(cat == "deep_shrink_underatten"))]
        picked = order[: args.n_per_cat].tolist()
        fig, axes = plt.subplots(1, len(picked), figsize=(6 * len(picked), 6))
        if len(picked) == 1:
            axes = [axes]
        for ax, i in zip(axes, picked):
            t, n = divmod(int(i), n_envs)
            info = (f"{cat}  seed{args.seed} env{n} f{t}\n"
                    f"clearance pol={float(c_pol[i]):.3f} tgt={float(c_tgt[i]):.3f} "
                    f"floor={float(c_floor[i]):.3f}\n"
                    f"smin pol={float(s_pol[i].min()):.2f} tgt={float(smin_tgt[i]):.2f}")
            draw_frame(ax, occ, df.cpu(), p_f[i], h_f[i], hexes_for(int(i)), info)
        out_path = os.path.join(OUT_DIR, f"sl_snapshots_seed{args.seed}_{cat}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"[snap] {cat}: saved {len(picked)} frame(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
