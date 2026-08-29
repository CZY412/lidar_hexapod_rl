#!/usr/bin/env python
"""Compare legacy rule-based envelope (envelope_adaptive) vs EA2 grid oracle.

Both methods run on the SAME occupancy grid and the SAME 10 Hz pose sequence:

* legacy ``compute_envelope_params`` is fed an idealised full-360-deg body
  point cloud simulated from the grid by raycasting (the inverted top lidar
  it was designed for; EA2 has no such sensor),
* EA2 ``compute_direct_oracle_params_with_stats`` uses the distance field
  directly (its idealised input).

Scenarios:
* static: open / corridor 0.65 / corridor 0.45 / front wall / rear pillar /
  side pillar  -- run to steady state, compare final params;
* dynamic: walk from open into a 0.65 m corridor and out -- compare per-frame
  params, collision frames, reaction latency, and per-frame param jumps
  (continuity).

Output: printed tables + PNG time-series under ``_outputs/``.

Run: python compare_envelope_legacy_vs_oracle.py
"""

from __future__ import annotations

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import os

import numpy as np
import torch

try:  # pytest: package ``ea2``
    from . import _ea2_testlib as tl
except ImportError:  # direct script execution
    import _ea2_testlib as tl

from legged_gym.envs.el_4090.envelope_adaptive.envelope_computer import (
    compute_envelope_params,
    _make_min_hex,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    hex_collision_terms,
)

# legacy sensor / rule settings (from envelope_adaptive/el_4090_ea_config.py)
_LEGACY_MAX_RANGE = 5.0
_LEGACY_N_AZIMUTH = 180
_LEGACY_Z_LEVELS = (-0.25, -0.05, 0.15)

_PARAM_NAMES = (
    "front_width",
    "middle_width",
    "back_width",
    "forward_limit",
    "backward_limit",
)

_DT = 0.1          # 10 Hz lidar cycle
_SPEED = 1.0       # m/s, same as EA2 stage 1

_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_outputs")


def _legacy_cfg():
    from types import SimpleNamespace

    return SimpleNamespace(
        envelope=SimpleNamespace(
            z_top=0.15,
            z_bottom=-0.25,
            margin_distance=0.25,
            hold_margin=0.1,
            shrink_step=0.03,
            grow_step=0.03,
            grow_cooldown_frames=5,
        ),
        commands=SimpleNamespace(
            ranges=SimpleNamespace(
                front_width=[0.3, 0.6],
                middle_width=[0.3, 0.7],
                back_width=[0.3, 0.6],
                forward_limit=[0.6, 0.9],
                backward_limit=[-0.9, -0.6],
            )
        ),
    )


_LEGACY = _legacy_cfg()


def _transition_mask(corridor_half_width: float) -> np.ndarray:
    """Open field with a corridor section for x in [0, 8]."""
    mask = np.zeros((tl.SIZE, tl.SIZE), dtype=bool)
    ys = np.arange(tl.SIZE) * tl.RES + tl.WORLD_MIN
    xs = np.arange(tl.SIZE) * tl.RES + tl.WORLD_MIN
    for ix, x in enumerate(xs):
        if 0.0 <= x <= 8.0:
            for iy, y in enumerate(ys):
                if abs(y) > corridor_half_width:
                    mask[iy, ix] = True
    return mask


def _legacy_pointcloud(pos_xy: np.ndarray, heading: float, mask: np.ndarray) -> torch.Tensor:
    """Idealised legacy sensor: full 360-deg raycast against the grid.

    Returns body-frame points (N, 3) at the legacy prism z-band.
    """
    az = np.linspace(0.0, 2.0 * np.pi, _LEGACY_N_AZIMUTH, endpoint=False)
    dirs = np.stack([np.cos(az + heading), np.sin(az + heading)], axis=-1)  # world frame
    ts = np.arange(0.05, _LEGACY_MAX_RANGE, 0.05, dtype=np.float32)

    px = pos_xy[0] + ts[None, :] * dirs[:, 0:1]  # (A, T)
    py = pos_xy[1] + ts[None, :] * dirs[:, 1:2]
    ix = ((px - tl.WORLD_MIN) / tl.RES).astype(np.int64)
    iy = ((py - tl.WORLD_MIN) / tl.RES).astype(np.int64)
    ok = (ix >= 0) & (ix < tl.SIZE) & (iy >= 0) & (iy < tl.SIZE)
    sxi = np.clip(ix, 0, tl.SIZE - 1)
    syi = np.clip(iy, 0, tl.SIZE - 1)
    hit = ok & mask[syi, sxi]
    first = np.argmax(hit, axis=1)  # first True along T, A axis kept
    has = hit.any(axis=1)

    tx = px[np.arange(len(az)), first]
    ty = py[np.arange(len(az)), first]
    tx = tx[has]
    ty = ty[has]

    # world -> body frame
    c, s = np.cos(-heading), np.sin(-heading)
    bx = c * (tx - pos_xy[0]) - s * (ty - pos_xy[1])
    by = s * (tx - pos_xy[0]) + c * (ty - pos_xy[1])

    pts = []
    for z in _LEGACY_Z_LEVELS:
        pts.append(np.stack([bx, by, np.full_like(bx, z)], axis=-1))
    cloud = np.concatenate(pts, axis=0).astype(np.float32)
    return torch.from_numpy(cloud).unsqueeze(0)  # (1, N, 3)


def _legacy_vec(params: dict) -> torch.Tensor:
    return torch.stack([params[name] for name in _PARAM_NAMES], dim=-1)


def run_legacy_sequence(
    poses: list,
    heading: float,
    mask: np.ndarray,
) -> torch.Tensor:
    """Run legacy rule computer frame by frame.  Returns (T, 5)."""
    min_hex = _make_min_hex(_LEGACY, device="cpu")
    params = {
        "front_width": torch.tensor([tl.HIGH[0]]),
        "middle_width": torch.tensor([tl.HIGH[1]]),
        "back_width": torch.tensor([tl.HIGH[2]]),
        "forward_limit": torch.tensor([tl.HIGH[3]]),
        "backward_limit": torch.tensor([tl.LOW[4]]),
    }
    cooldown = torch.zeros(1, 5, dtype=torch.int64)
    out = []
    # settle at the first pose so static scenarios reach steady state
    for i, (x, y) in enumerate(poses):
        pts = _legacy_pointcloud(np.array([x, y]), heading, mask)
        params, cooldown = compute_envelope_params(
            pts,
            torch.zeros(1, 3),
            torch.zeros(1, 4),
            _LEGACY,
            params,
            cooldown,
            min_hex,
        )
        out.append(_legacy_vec(params)[0].clone())
    return torch.stack(out, dim=0)


def run_oracle_sequence(
    poses: list,
    heading: float,
    df: np.ndarray,
    interp_crossing: bool = False,
) -> torch.Tensor:
    pos = torch.tensor(poses, dtype=torch.float32)
    head = torch.full((len(poses),), heading)
    return tl.oracle_batch(head, pos, df, interp=interp_crossing, from_numpy=True)


def _hard_collision(seq: torch.Tensor, heading: float, poses, df: np.ndarray) -> np.ndarray:
    dft = torch.as_tensor(df, dtype=torch.float32)
    out = []
    for t, (x, y) in enumerate(poses):
        _, hard = hex_collision_terms(
            seq[t : t + 1],
            torch.tensor([heading]),
            torch.tensor([[x, y]], dtype=torch.float32),
            dft,
            margin=tl.MARGIN,
            soft_margin=tl.SOFT_MARGIN,
        )
        out.append(float(hard[0]) > 1e-3)
    return np.array(out)


def _print_seq_stats(name: str, seq: torch.Tensor, unsafe: np.ndarray) -> None:
    print(
        f"  {name:22s} unsafe_frames={unsafe.mean():6.2f} "
        f"potential(mean)={tl.potential(seq).mean():.3f} "
        f"max_frame_jump={tl.max_frame_jump(seq):.3f}"
    )


def scenario_static() -> None:
    print("=" * 78)
    print("STATIC SCENARIOS (legacy settles 200 frames @10Hz; oracle stateless)")
    heading = 0.0
    cases = []

    df, mask = tl.corridor_field(2.0)
    cases.append(("open 2.0m", df, mask))
    df, mask = tl.corridor_field(0.65)
    cases.append(("corridor 0.65m", df, mask))
    df, mask = tl.corridor_field(0.45)
    cases.append(("corridor 0.45m", df, mask))

    df, mask = tl.point_field([(0.85, 0.0)])
    cases.append(("front pillar 0.85", df, mask))
    df, mask = tl.point_field([(-0.85, 0.0)])
    cases.append(("rear pillar 0.85", df, mask))
    df, mask = tl.point_field([(0.0, 0.85)])
    cases.append(("side pillar 0.85", df, mask))

    header = (
        f"  {'case':22s} {'method':8s} "
        + " ".join(f"{n[:9]:>9s}" for n in _PARAM_NAMES)
        + "    unsafe  potential"
    )
    print(header)
    for name, df, mask in cases:
        poses = [(0.0, 0.0)] * 200
        leg = run_legacy_sequence(poses, heading, mask)[-1:]
        orc = run_oracle_sequence([(0.0, 0.0)], heading, df)
        orc_i = run_oracle_sequence([(0.0, 0.0)], heading, df, interp_crossing=True)
        for tag, seq in (
            ("legacy", leg),
            ("ea2-oracle", orc),
            ("oracle+interp", orc_i),
        ):
            _, hard = hex_collision_terms(
                seq,
                torch.tensor([heading]),
                torch.tensor([[0.0, 0.0]], dtype=torch.float32),
                torch.as_tensor(df, dtype=torch.float32),
                margin=tl.MARGIN,
                soft_margin=tl.SOFT_MARGIN,
            )
            p = seq[0].numpy()
            print(
                f"  {name:22s} {tag:8s} "
                + " ".join(f"{v:9.4f}" for v in p)
                + f"  {float(hard[0]) > 1e-3!s:>6}  {tl.potential(seq).item():.3f}"
            )
    print()


def scenario_dynamic():
    print("=" * 78)
    print("DYNAMIC SCENARIO: walk +x from open into 0.65m corridor (x in [0,8]) and out")
    heading = 0.0
    mask = _transition_mask(0.65)
    df = tl.distance_field(mask)

    xs = np.arange(-6.0, 14.0 + 1e-9, _SPEED * _DT)  # 200 frames, 20 s
    poses = [(float(x), 0.0) for x in xs]

    leg = run_legacy_sequence(poses, heading, mask)
    orc = run_oracle_sequence(poses, heading, df)
    orc_i = run_oracle_sequence(poses, heading, df, interp_crossing=True)
    orc_rl = tl.apply_rate_limit(
        orc_i, tl.RateLimitedOracle(shrink_step=0.03, grow_step=0.03)
    )

    leg_unsafe = _hard_collision(leg, heading, poses, df)
    orc_unsafe = _hard_collision(orc, heading, poses, df)
    orci_unsafe = _hard_collision(orc_i, heading, poses, df)
    orcr_unsafe = _hard_collision(orc_rl, heading, poses, df)

    print(f"  frames={len(poses)} (200 = 20 s @10Hz, 1 m/s)")
    _print_seq_stats("legacy(360 lidar)", leg, leg_unsafe)
    _print_seq_stats("ea2-oracle(grid)", orc, orc_unsafe)
    _print_seq_stats("oracle+interp", orc_i, orci_unsafe)
    _print_seq_stats("interp+ratelimit", orc_rl, orcr_unsafe)

    # Both methods perceive 5 m ahead, so the meaningful quantities are
    # anticipation distance (shrink starts before the walls at x=0) and
    # recovery distance (still shrunk after the walls end at x=8), plus
    # smoothness (second-difference energy of middle_width).
    def _anticipate(seq: torch.Tensor) -> float:
        ref = float(seq[5, 1])  # middle_width deep in open area
        for i in range(len(seq)):
            if float(seq[i, 1]) < ref - 0.02:
                return -xs[i]  # metres before wall start (x=0)
        return float("nan")

    def _recover(seq: torch.Tensor) -> float:
        ref = float(seq[5, 1])
        for i in range(len(seq) - 1, -1, -1):
            if float(seq[i, 1]) < ref - 0.02:
                return xs[i] - 8.0  # metres after wall end (x=8)
        return float("nan")

    def _smoothness(seq: torch.Tensor) -> float:
        mw = seq[:, 1]
        return float((mw[2:] - 2 * mw[1:-1] + mw[:-2]).abs().mean())

    for tag, seq in (
        ("legacy", leg),
        ("oracle", orc),
        ("o+interp", orc_i),
        ("interp+rl", orc_rl),
    ):
        print(
            f"  {tag:10s} anticipate={_anticipate(seq):5.2f} m before walls  "
            f"recover={_recover(seq):5.2f} m after walls  "
            f"mw 2nd-diff={_smoothness(seq):.5f}"
        )
    print()

    return xs, {
        "legacy": leg,
        "oracle": orc,
        "oracle_interp": orc_i,
        "interp_rl": orc_rl,
    }


def plot_dynamic(xs, seqs, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    series = [
        ("legacy (360 lidar)", seqs["legacy"]),
        ("EA2 grid oracle (raw)", seqs["oracle"]),
        ("oracle+interp+ratelimit", seqs["interp_rl"]),
    ]
    for i, ax in enumerate(axes):
        for label, seq in series:
            ax.plot(xs, seq[:, i], label=label, lw=1.5)
        ax.axvspan(0, 8, color="0.85", zorder=0)
        ax.set_ylabel(_PARAM_NAMES[i])
        if i == 0:
            ax.legend(loc="lower left", fontsize=8)
            ax.set_title("Legacy rule-based vs EA2 grid oracle envelope (0.65m corridor, x in [0,8])")
    axes[-1].set_xlabel("x position (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  plot saved: {path}")


def main() -> None:
    np.set_printoptions(suppress=True)
    scenario_static()
    xs, seqs = scenario_dynamic()
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    plot_dynamic(xs, seqs, os.path.join(_OUTPUT_DIR, "compare_envelope_legacy_vs_oracle.png"))


if __name__ == "__main__":
    main()
