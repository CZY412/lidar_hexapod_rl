"""Performance benchmark for the envelope oracle marches.

NOT collected by pytest (the filename does not start with ``test_``), because
absolute timings on a shared GPU are too noisy for a pass/fail gate.  Run it
manually:

    conda activate el4090
    cd legged_gym/legged_gym
    python legged_gym/tests/ea2/bench_oracle.py

It reports the per-call cost of the axis and coupled marches across batch
sizes.  Because the workload is host-dispatch bound (not FLOP bound) the time
is nearly independent of the batch size -- if that ever stops being true, the
optimisation assumptions have been violated.
"""

from __future__ import annotations

import sys
import time

import isaacgym  # noqa: F401  must be imported before torch

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, ".")

from legged_gym.envs.el_4090.envelope_adaptive_2 import envelope_oracle as eo

_LOW = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9], dtype=torch.float32)
_HIGH = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6], dtype=torch.float32)
_RES = 0.1
_SIZE = 740
_WMIN = -37.0
_MARGIN = 0.10
_STEP = 0.05
_MAX_DIST = 5.0


def make_field(seed: int = 0, n_tiles: int = 4, per_tile: int = 18) -> torch.Tensor:
    """Production-shaped pillar field (740x740) on the default device."""
    rng = np.random.default_rng(seed)
    boxes, min_sep = [], 2.6
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for tx in range(n_tiles):
        for ty in range(n_tiles):
            ox, oy = -32.0 + tx * 16.0, -32.0 + ty * 16.0
            placed, attempts = 0, 0
            while placed < per_tile and attempts < 500:
                attempts += 1
                cx = ox + 1.0 + rng.random() * 14.0
                cy = oy + 1.0 + rng.random() * 14.0
                a = 0.5 + rng.random() * 3.5
                b = 0.5 + rng.random() * 3.5
                hx, hy = (a / 2, b / 2) if rng.random() < 0.5 else (b / 2, a / 2)
                if all(abs(cx - bx) > hx + bhx + min_sep or
                       abs(cy - by) > hy + bhy + min_sep for bx, by, bhx, bhy in boxes):
                    boxes.append((cx, cy, hx, hy))
                    placed += 1
    xs = np.arange(_SIZE) * _RES + _WMIN
    XX, YY = np.meshgrid(xs, xs)
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    for cx, cy, hx, hy in boxes:
        mask |= (np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)
    df = ndimage.distance_transform_edt(~mask, sampling=(_RES, _RES)).astype(np.float32)
    return torch.from_numpy(df).to(dev)


def realistic_poses(df, n, seed=3, clearance=0.35):
    """Poses on the inflated-free space, i.e. where an A* path would run."""
    dev = df.device
    g = torch.Generator().manual_seed(seed)
    out, got = [], 0
    while got < n:
        cand = (torch.rand(n * 4, 2, generator=g) * 64.0 - 32.0).to(dev)
        ix = torch.floor((cand[:, 0] - _WMIN) / _RES).long().clamp(0, _SIZE - 1)
        iy = torch.floor((cand[:, 1] - _WMIN) / _RES).long().clamp(0, _SIZE - 1)
        sel = cand[df[iy, ix] >= clearance]
        out.append(sel)
        got += int(sel.shape[0])
    pos = torch.cat(out)[:n].contiguous()
    head = (torch.rand(n, generator=g) * 6.283185 - 3.141592).to(dev).contiguous()
    return head, pos


def timeit(fn, iters=15, warmup=5):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    best = None
    for _ in range(3):
        if torch.cuda.is_available():
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            t = s.elapsed_time(e) / iters
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            t = (time.perf_counter() - t0) / iters * 1000.0
        best = t if best is None else min(best, t)
    return best


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {dev}")
    if torch.cuda.is_available():
        print(f"gpu    = {torch.cuda.get_device_name(0)}")
    df = make_field()
    low, high = _LOW.to(dev), _HIGH.to(dev)
    print(f"field  = {tuple(df.shape)}\n")

    print(f"{'N':>6}  {'axis (ms)':>11}  {'coupled (ms)':>13}")
    print("-" * 36)
    for N in (64, 256, 1024, 2048):
        try:
            h, p = realistic_poses(df, N, seed=3)
        except RuntimeError as ex:
            print(f"{N:>6}  skipped ({str(ex).split('.')[0][:40]})")
            continue
        t_axis = timeit(lambda: eo._compute_raw_scales(
            h, p, df, low, high, _MARGIN, _STEP, _MAX_DIST, True, "axis"))
        t_cpl = timeit(lambda: eo._compute_raw_scales(
            h, p, df, low, high, _MARGIN, _STEP, _MAX_DIST, True, "coupled"))
        print(f"{N:>6}  {t_axis:11.3f}  {t_cpl:13.3f}")
        del h, p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nNote: the march is host-dispatch bound, so timings should be nearly")
    print("flat in N.  A strong N-dependence means the batched implementation")
    print("has regressed into a memory-bound one (check the t-axis chunking).")


if __name__ == "__main__":
    main()
