#!/usr/bin/env python
"""Smoothness + correctness validation of the EA2 oracle over full envelope
deformation processes (approach / lateral pass / in-place rotation / free tour
through a pillar field).

Per scenario x method (raw / +interp / +interp+ratelimit):

* feasibility   -- a frame is INFEASIBLE when even the minimum envelope
                   (all extents at their hard bounds) collides; oracle
                   methods are only accountable on feasible frames.  Reported
                   as ``infeas%`` (method independent).
* safety        -- worst boundary+interior sample clearance over FEASIBLE
                   frames, computed independently from ``_hex_sample_violations``
                   (clearance = margin - violation * soft_margin).
* smoothness    -- max per-frame parameter jump and mean |2nd difference|;
                   the rate-limited variant must respect the configured
                   shrink bound on non-snapped frames.
* tightness     -- over feasible frames: every non-saturated parameter
                   (extent scale < 0.98) has a boundary group sample within
                   6 cm of the margin contour (shrink is justified).

Run: python validate_oracle_smoothness.py
"""

from __future__ import annotations

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import numpy as np
import torch

try:  # pytest: package ``ea2``
    from . import _ea2_testlib as tl
except ImportError:  # direct script execution
    import _ea2_testlib as tl

from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    _hex_sample_violations,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
    _DIRECT_BOUNDARY_GROUPS,
    _physical_min_max,
)

_DT = 0.1  # 10 Hz
# Shrink rate must keep up with the fastest approach speed: at 1 m/s and 10 Hz
# the demanded shrink is 0.10 m/frame, so 0.03 (legacy default for a slow
# robot) lets the envelope penetrate obstacles.  Growth lag is only
# conservatism, not danger, so it stays slow.
_RL_SHRINK = 0.12
_RL_GROW = 0.03

_MIN_V, _MAX_V = _physical_min_max(tl.LOW, tl.HIGH)
_SPAN = _MAX_V - _MIN_V  # signed


def _tour_poses(mask: np.ndarray, n_frames: int = 600):
    """Reactive tour: drive forward, rotate in place when blocked ahead.

    The ahead-clearance gate (0.6 m) keeps the tour near obstacles while the
    forced 90-degree turns every 50 frames exercise rotation deformation.
    """
    dft = tl.distance_field(mask)

    def clear_at(x: float, y: float, need: float) -> bool:
        ix, iy = tl.grid_index(np.array([x]), np.array([y]))
        return dft[iy[0], ix[0]] >= need

    start = None
    for x in np.arange(-30.0, 30.0, 0.5):
        if clear_at(float(x), -24.0, 1.0):
            start = (float(x), -24.0, 0.0)
            break
    assert start is not None

    poses = [start]
    x, y, h = start
    turn_left = 0
    for i in range(n_frames - 1):
        lx = x + 0.1 * np.cos(h)
        ly = y + 0.1 * np.sin(h)
        ax = x + 0.8 * np.cos(h)
        ay = y + 0.8 * np.sin(h)
        if (i + 1) % 50 == 0 and turn_left == 0:
            turn_left = 11  # 90 deg at omega_max over ~11 frames
        if turn_left > 0:
            h += 1.5 * _DT  # rotate in place at omega_max
            turn_left -= 1
        elif clear_at(lx, ly, 0.6) and clear_at(ax, ay, 0.6):
            x, y = lx, ly
        else:
            h += 1.5 * _DT  # blocked: rotate in place
        poses.append((x, y, h))
    return poses


def _run_methods(df: np.ndarray, poses):
    pos = torch.tensor([(q[0], q[1]) for q in poses], dtype=torch.float32)
    head = torch.tensor([q[2] for q in poses], dtype=torch.float32)
    dft = torch.as_tensor(df, dtype=torch.float32)

    raw = tl.oracle_batch(head, pos, dft, interp=False)
    itp = tl.oracle_batch(head, pos, dft, interp=True)

    # _RL_SHRINK/_RL_GROW are extent/call at 10 Hz -> per-second rates
    rl = tl.RateLimitedOracle(
        num_envs=1, dt=_DT, device="cpu", low=tl.LOW, high=tl.HIGH,
        shrink_rate=_RL_SHRINK / _DT, grow_rate=_RL_GROW / _DT,
        cooldown_seconds=0.5,
    )
    seq_frames = []
    snap_frames = []
    for i in range(len(poses)):
        head_i = head[i : i + 1]
        pos_i = pos[i : i + 1]

        def check(cand: torch.Tensor, _h=head_i, _p=pos_i) -> torch.Tensor:
            viol = _hex_sample_violations(
                cand, _h, _p, dft, margin=tl.MARGIN, soft_margin=tl.SOFT_MARGIN
            )
            return torch.tensor([float(viol.max()) > 0.05], dtype=torch.bool)

        rl.safety_check = check
        out = rl.update(itp[i : i + 1])[0]
        seq_frames.append(out)
        snap_frames.append(bool(rl.snapped[0]))
    rl_seq = torch.stack(seq_frames, dim=0)
    return {"raw": raw, "interp": itp, "interp+rl": rl_seq}, torch.tensor(snap_frames)


def _extent_scales(params: torch.Tensor) -> torch.Tensor:
    return ((params - _MIN_V) / _SPAN).clamp(0.0, 1.0)


def _metrics(df: np.ndarray, poses, seq: torch.Tensor, dft, snap=None) -> dict:
    pos = torch.tensor([(q[0], q[1]) for q in poses], dtype=torch.float32)
    head = torch.tensor([q[2] for q in poses], dtype=torch.float32)

    clearance = tl.sample_clearances(seq, head, pos, dft)
    min_per_frame = clearance.min(dim=-1).values  # (T,)

    # feasibility from the minimum envelope (method independent)
    min_env_clear = tl.sample_clearances(
        tl.min_envelope(tl.LOW, tl.HIGH).to(seq.device).expand(len(poses), -1),
        head,
        pos,
        dft,
    ).min(dim=-1).values
    feasible = min_env_clear >= tl.MARGIN - 1e-3

    n_infeas = int((~feasible).sum())
    if feasible.any():
        worst = float(min_per_frame[feasible].min())
        unsafe = worst < tl.MARGIN - 1e-3
        bad_frames = (
            (min_per_frame < tl.MARGIN - 1e-3) & feasible
        ).nonzero().flatten().tolist()
    else:
        worst, unsafe, bad_frames = float("nan"), False, []

    if snap is not None and len(seq) > 1:
        jumps = (seq[1:] - seq[:-1]).abs().max(dim=-1).values
        keep = ~snap[1:]
        jump = float(jumps[keep].max()) if bool(keep.any()) else 0.0
    else:
        jump = tl.max_frame_jump(seq)
    smooth = tl.second_difference(seq)

    # tightness over feasible frames
    scales = _extent_scales(seq)  # (T, 5)
    group_names = list(_DIRECT_BOUNDARY_GROUPS.keys())
    tight_frames = 0
    counted = 0
    for t in range(len(seq)):
        if not bool(feasible[t]):
            continue
        counted += 1
        ok = True
        for j, name in enumerate(group_names):
            if scales[t, j] < 0.98:
                c = clearance[t, _DIRECT_BOUNDARY_GROUPS[name]]
                if float(c.min()) > tl.MARGIN + 0.06:
                    ok = False
                    break
        tight_frames += int(ok)
    tight = tight_frames / max(counted, 1)

    return {
        "infeas%": 100.0 * n_infeas / len(poses),
        "min_clear_feas": worst,
        "unsafe": unsafe,
        "bad_frames": bad_frames[:8],
        "max_jump": jump,
        "smooth": smooth,
        "tight": tight,
    }


def _report(name: str, poses, seqs: dict, dft, snap=None) -> None:
    print(f"\n{'=' * 88}\n{name}  ({len(poses)} frames @10Hz)\n{'=' * 88}")
    print(
        f"  {'method':12s} {'infeas%':>8s} {'min_clr_feas':>12s} {'unsafe':>7s} "
        f"{'max_jump':>9s} {'2nd_diff':>9s} {'tight':>7s}"
    )
    for tag, seq in seqs.items():
        m = _metrics(None, poses, seq, dft, snap if tag == "interp+rl" else None)
        print(
            f"  {tag:12s} {m['infeas%']:8.1f} {m['min_clear_feas']:12.4f} "
            f"{str(m['unsafe']):>7s} {m['max_jump']:9.4f} {m['smooth']:9.5f} "
            f"{m['tight']:7.2%}"
        )
        if m["bad_frames"] and tag == "raw":
            print(f"             first unsafe frames (raw): {m['bad_frames']}")
        if tag == "interp+rl":
            assert m["max_jump"] <= _RL_SHRINK + 1e-4, f"rate limit violated: {m['max_jump']}"


def main() -> None:
    # A: head-on approach to a thin wall
    df, _ = tl.wall_field()
    dft = torch.as_tensor(df, dtype=torch.float32)
    poses = [(x, 0.0, 0.0) for x in np.arange(4.0, 0.80, -0.05)]
    _report("A. head-on approach to thin wall (stop at 0.8m)", poses, _run_methods(df, poses)[0], dft)

    # B: lateral pass by a 2x2 pillar, 1.6 m centre offset (feasible for min env)
    df, _ = tl.pillar_field_2x2()
    dft = torch.as_tensor(df, dtype=torch.float32)
    poses = [(x, 1.6, 0.0) for x in np.arange(-5.0, 5.0 + 1e-9, 0.1)]
    seqs, snap = _run_methods(df, poses)
    _report("B. lateral pass 2x2 pillar @1.6m offset", poses, seqs, dft, snap)

    # C1: in-place rotation beside a thin wall (0.9 m gap), full turn
    df, _ = tl.wall_field()
    dft = torch.as_tensor(df, dtype=torch.float32)
    poses = [(0.9, 0.0, h) for h in np.arange(0.0, 2 * np.pi, 1.5 * _DT)]
    seqs, snap = _run_methods(df, poses)
    _report("C1. rotation 0.9m beside thin wall", poses, seqs, dft, snap)

    # C2: in-place rotation centred in a 2.0 m corridor
    df, _ = tl.corridor_field(1.0)
    dft = torch.as_tensor(df, dtype=torch.float32)
    poses = [(0.0, 0.0, h) for h in np.arange(0.0, 2 * np.pi, 1.5 * _DT)]
    seqs, snap = _run_methods(df, poses)
    _report("C2. rotation centred in 2.0m corridor", poses, seqs, dft, snap)

    # D: reactive tour through a random 4x4-tile pillar field
    df, mask = tl.random_pillar_field(seed=0)
    dft = torch.as_tensor(df, dtype=torch.float32)
    poses = _tour_poses(mask, n_frames=600)
    seqs, snap = _run_methods(df, poses)
    n_snap = int(snap.sum())
    print(f"  (interp+rl safety snaps: {n_snap}/{len(poses)} frames)")
    _report("D. reactive tour in random pillar field", poses, seqs, dft, snap)

    print(f"\nrate-limited jump bound asserted (<= {_RL_SHRINK} + 1e-4) for all scenarios")


if __name__ == "__main__":
    main()
