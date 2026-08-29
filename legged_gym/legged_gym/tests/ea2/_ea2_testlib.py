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
):
    """Chunked ``compute_direct_oracle_params_with_stats`` over many poses."""
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
# rate-limited oracle target smoother
# ---------------------------------------------------------------------------

class RateLimitedOracle:
    """Shrink-fast / grow-slow / cooldown post-filter on the oracle target.

    Mirrors the legacy rule computer's hysteresis but in normalised extent
    space, applied as post-processing to the stateless grid oracle.  An
    optional ``safety_check(candidate) -> bool`` snaps the frame straight to
    the raw (safety-verified) oracle when the rate-limited candidate would
    itself be unsafe.
    """

    def __init__(
        self,
        low: torch.Tensor = LOW,
        high: torch.Tensor = HIGH,
        shrink_step: float = 0.03,
        grow_step: float = 0.03,
        cooldown: int = 5,
        grow_tol_frac: float = 0.5,
        safety_check=None,
    ):
        from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import _physical_min_max

        self.min_v, self.max_v = _physical_min_max(low, high)
        # Signed span: backward_limit is physically reversed (more negative =
        # larger rear extent), so its span is negative.  Division by the
        # signed span makes s=0 fully shrunk and s=1 fully extended for ALL
        # five parameters; rate steps divide by |span|.
        self.span = self.max_v - self.min_v
        abs_span = self.span.abs().clamp_min(1e-6)
        self.shrink_n = shrink_step / abs_span
        self.grow_n = grow_step / abs_span
        self.grow_tol = grow_tol_frac * self.grow_n
        self.cooldown = cooldown
        self.safety_check = safety_check
        self.last_snapped = False
        self.prev_s: torch.Tensor | None = None
        self.counter: torch.Tensor | None = None

    def reset(self) -> None:
        self.prev_s = None
        self.counter = None

    def _to_s(self, params: torch.Tensor) -> torch.Tensor:
        return ((params - self.min_v) / self.span).clamp(0.0, 1.0)

    def _from_s(self, s: torch.Tensor) -> torch.Tensor:
        return self.min_v + s * self.span

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        raw_s = self._to_s(params)
        if self.prev_s is None:
            self.prev_s = torch.ones_like(raw_s)  # start fully open
            self.counter = torch.zeros_like(raw_s)
        needs_shrink = raw_s < self.prev_s - 1e-6
        clear = raw_s > self.prev_s + self.grow_tol
        self.counter = torch.where(
            needs_shrink,
            torch.zeros_like(self.counter),
            torch.where(clear, self.counter + 1, self.counter),
        )
        shrink_target = (self.prev_s - self.shrink_n).clamp(0.0, 1.0)
        grow_target = (self.prev_s + self.grow_n).clamp(0.0, 1.0)
        can_grow = clear & (self.counter >= self.cooldown)
        new_s = torch.where(
            needs_shrink,
            torch.maximum(raw_s, shrink_target),
            torch.where(can_grow, torch.minimum(raw_s, grow_target), self.prev_s),
        )
        self.prev_s = new_s
        candidate = self._from_s(new_s)
        self.last_snapped = False
        if self.safety_check is not None and bool(self.safety_check(candidate)):
            self.last_snapped = True
            self.prev_s = raw_s
            return params
        return candidate


def apply_rate_limit(raw_seq: torch.Tensor, rl: "RateLimitedOracle") -> torch.Tensor:
    """Apply a RateLimitedOracle frame by frame over a (T, 5) sequence."""
    return torch.stack(
        [rl(raw_seq[i : i + 1])[0] for i in range(raw_seq.shape[0])], dim=0
    )
