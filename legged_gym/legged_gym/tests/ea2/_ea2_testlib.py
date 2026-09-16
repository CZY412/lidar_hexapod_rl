"""Shared helpers for EA2 test scripts and unit tests.

Single source of truth for the duplicated fixtures that previously lived in
compare_envelope_legacy_vs_oracle.py / validate_*.py: frozen envelope bounds,
grid constants, synthetic distance fields, the chunked oracle invocation,
sample-clearance evaluation, minimum-envelope feasibility, and the
RateLimitedOracle target smoother.

Import shim: scripts run directly (``python tests/ea2/x.py``) put this
directory on ``sys.path[0]`` while pytest imports them as ``ea2.x``; support
both.  Production code must NOT import from here.
"""

from __future__ import annotations

import isaacgym  # noqa: F401  (must precede torch via legged_gym imports)

import numpy as np
import torch
from scipy import ndimage

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts

# Grid constants: single source of truth is the frozen contract module.
WORLD_MIN = _contracts.EA2_WORLD_MIN_XY
RES = _contracts.EA2_RESOLUTION_M
SIZE = _contracts.EA2_GRID_SHAPE[0]
assert _contracts.EA2_GRID_SHAPE[0] == _contracts.EA2_GRID_SHAPE[1], "square grid assumed"

# Frozen envelope parameter bounds (spider_envelop contract, README 2.2.3).
# Pinned against the loaded spec by test_contracts.py.
LOW = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9], dtype=torch.float32)
HIGH = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6], dtype=torch.float32)

# Default oracle / collision semantics used across the scripts.
MARGIN = 0.10
SOFT_MARGIN = 0.10

# Default oracle / collision semantics used across the scripts.
MARGIN = 0.10
SOFT_MARGIN = 0.10


# ---------------------------------------------------------------------------
# distance fields
# ---------------------------------------------------------------------------

def distance_field(mask: np.ndarray) -> np.ndarray:
    """Unsigned distance-to-obstacle field (metres) for an occupancy mask."""
    return ndimage.distance_transform_edt(
        ~mask, sampling=(RES, RES)
    ).astype(np.float32)


def grid_index(wx, wy):
    wx = np.asarray(wx)
    wy = np.asarray(wy)
    ix = np.round((wx - WORLD_MIN) / RES).astype(np.int64)
    iy = np.round((wy - WORLD_MIN) / RES).astype(np.int64)
    return ix, iy


def corridor_field(half_width: float):
    """Walls at |y| > half_width everywhere -> (distance field, mask)."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    ys = np.arange(SIZE) * RES + WORLD_MIN
    for iy, y in enumerate(ys):
        if abs(y) > half_width:
            mask[iy, :] = True
    return distance_field(mask), mask


def point_field(pillars) -> tuple[np.ndarray, np.ndarray]:
    """Single-cell pillars at the given (x, y) world positions."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for wx, wy in pillars:
        ix, iy = grid_index(np.array([wx]), np.array([wy]))
        mask[iy[0], ix[0]] = True
    return distance_field(mask), mask


def wall_field() -> tuple[np.ndarray, np.ndarray]:
    """Thin occupied column at x = 0 (free on both sides)."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    ix, _ = grid_index(np.array([0.0]), np.array([0.0]))
    mask[:, ix[0]] = True
    return distance_field(mask), mask


def pillar_field_2x2() -> tuple[np.ndarray, np.ndarray]:
    """Single 2m x 2m pillar centred at the origin."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    ix0, iy0 = grid_index(np.array([-1.0]), np.array([-1.0]))
    ix1, iy1 = grid_index(np.array([1.0]), np.array([1.0]))
    mask[iy0[0] : iy1[0] + 1, ix0[0] : ix1[0] + 1] = True
    return distance_field(mask), mask


def random_pillar_field(seed: int = 0):
    """4x4 tiles x 18 pillars, config-like sizes/separation (AABB rejection)."""
    rng = np.random.default_rng(seed)
    boxes: list[tuple[float, float, float, float]] = []  # cx, cy, hx, hy
    min_sep = 2.6
    for tx in range(4):
        for ty in range(4):
            ox, oy = -32.0 + tx * 16.0, -32.0 + ty * 16.0
            placed = 0
            attempts = 0
            while placed < 18 and attempts < 500:
                attempts += 1
                cx = ox + 1.0 + rng.random() * 14.0
                cy = oy + 1.0 + rng.random() * 14.0
                long_side = 0.5 + rng.random() * 3.5
                short_side = 0.5 + rng.random() * 3.5
                if rng.random() < 0.5:
                    hx, hy = long_side / 2, short_side / 2
                else:
                    hx, hy = short_side / 2, long_side / 2
                ok = True
                for bx, by, bhx, bhy in boxes:
                    if (
                        abs(cx - bx) < hx + bhx + min_sep
                        and abs(cy - by) < hy + bhy + min_sep
                    ):
                        ok = False
                        break
                if ok:
                    boxes.append((cx, cy, hx, hy))
                    placed += 1

    mask = np.zeros((SIZE, SIZE), dtype=bool)
    xs = np.arange(SIZE) * RES + WORLD_MIN
    ys = np.arange(SIZE) * RES + WORLD_MIN
    XX, YY = np.meshgrid(xs, ys)
    for cx, cy, hx, hy in boxes:
        mask |= (np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)
    return distance_field(mask), mask


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def potential(params5: torch.Tensor) -> torch.Tensor:
    """Normalized-parameter mean potential with the backward-limit reversal."""
    low5 = LOW.to(params5.device)
    high5 = HIGH.to(params5.device)
    norm = (params5 - low5) / (high5 - low5).clamp_min(1e-6)
    norm = norm.clone()
    norm[..., 4] = (-params5[..., 4] - (-high5[4])) / (
        (-low5[4]) - (-high5[4])
    ).clamp_min(1e-6)
    return norm.clamp(0.0, 1.0).mean(dim=-1)


def max_frame_jump(seq: torch.Tensor) -> float:
    if len(seq) < 2:
        return 0.0
    return float((seq[1:] - seq[:-1]).abs().max())


def second_difference(seq: torch.Tensor) -> float:
    if len(seq) < 3:
        return 0.0
    return float((seq[2:] - 2 * seq[1:-1] + seq[:-2]).abs().mean())


# ---------------------------------------------------------------------------
# oracle invocation + independent safety evaluation
# ---------------------------------------------------------------------------

#: Production group mode.  ``envelope.oracle_group_mode`` in
#: ``el_4090_ea2_config``; kept in sync so shared helpers and diagnostic
#: scripts exercise the same path the trainer actually uses.
DEFAULT_GROUP_MODE = "axis"


def oracle_batch(
    head: torch.Tensor,
    pos: torch.Tensor,
    df,
    low: torch.Tensor = LOW,
    high: torch.Tensor = HIGH,
    *,
    interp: bool,
    chunk: int = 64,
    max_dist: float = 5.0,
    from_numpy: bool = False,
    group_mode: str = DEFAULT_GROUP_MODE,
):
    """Chunked ``compute_direct_oracle_params_with_stats`` over many poses.

    Defaults to the production ``group_mode`` (``axis``) so diagnostic scripts
    measure the behaviour the trainer actually sees.  Pass
    ``group_mode="coupled"`` to exercise the legacy shared-boundary-group path.
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import compute_direct_oracle_params_with_stats

    if from_numpy:
        df = np.asarray(df)
    dft = df if isinstance(df, torch.Tensor) else torch.as_tensor(df, dtype=torch.float32)
    out = []
    for i in range(0, head.shape[0], chunk):
        params, _ = compute_direct_oracle_params_with_stats(
            head[i : i + chunk],
            pos[i : i + chunk],
            dft,
            low,
            high,
            margin=MARGIN,
            step=0.05,
            max_dist=max_dist,
            interp_crossing=interp,
            group_mode=group_mode,
        )
        out.append(params)
    return torch.cat(out, dim=0)


def sample_clearances(seq, head, pos, dft, margin: float = MARGIN, soft: float = SOFT_MARGIN):
    """Per-sample clearance (metres) via the production violation model.

    Clearance = margin - violation * soft_margin; independent of the oracle,
    so it is a valid safety oracle for envelope params.
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import _hex_sample_violations

    viol = _hex_sample_violations(seq, head, pos, dft, margin=margin, soft_margin=soft)
    return margin - viol * soft


def min_envelope(low: torch.Tensor = LOW, high: torch.Tensor = HIGH) -> torch.Tensor:
    """Fully shrunk envelope (all extents at their hard bounds)."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import _physical_min_max

    min_v, _ = _physical_min_max(low, high)
    return min_v


def feasible_mask(seq, head, pos, dft, low=LOW, high=HIGH, tol: float = 1e-3):
    """Boolean per-frame mask: the minimum envelope fits (clearance ok)."""
    min_env = min_envelope(low, high).to(seq.device)
    clr = sample_clearances(min_env.expand(seq.shape[0], -1), head, pos, dft)
    return clr[:, :24].min(-1).values >= MARGIN - tol
# ---------------------------------------------------------------------------
# rate-limited oracle target smoother (production implementation)
# ---------------------------------------------------------------------------

from legged_gym.envs.el_4090.envelope_adaptive_2.target_smoother import (
    RateLimitedOracle,
)


def apply_rate_limit(raw_seq: torch.Tensor, rl: RateLimitedOracle) -> torch.Tensor:
    """Apply a RateLimitedOracle frame by frame over a (T, 5) sequence."""
    return torch.stack(
        [rl.update(raw_seq[i : i + 1])[0] for i in range(raw_seq.shape[0])], dim=0
    )
