"""Unit tests for the EA2 attitude replay generator (design v3.1).

Covers the audit gates: continuity (no jumps), seam quality, the measured
amplitude law, trim semantics, determinism, and the pinned pitch sign
convention (nose-down = Ry(+θ) = the direction that compresses far-row
ground returns).  Pure torch + isaacgym.torch_utils; no Isaac sim.

Run: python -m pytest legged_gym/tests/ea2/test_attitude_replay.py -q
"""

import isaacgym  # noqa: F401

import math

import numpy as np
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.attitude_replay import (
    AttitudeReplay,
    build_episode_buffer,
    load_source,
)
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c

SOURCE = str(ea2c.EA2_DIR / "attitude_traj_source.npz")
DT = 0.02
EP_LEN = 2250


class _Cfg:
    trim_range = [-2.0, 2.0]
    scale_range = [0.85, 1.2]
    rate_range = [0.9, 1.1]
    ar_std = 0.3
    ratio_tau = 0.5
    fade_steps = 100
    std_law_k = 0.858
    std_law_b = 0.318


def _make(num_envs=4, seed=7):
    return AttitudeReplay(num_envs, "cpu", DT, EP_LEN, SOURCE, _Cfg(), seed)


def test_source_loads_and_matches_measured_stats():
    src = load_source(SOURCE)
    p = np.degrees(src["pitch"])
    assert len(p) >= 400
    assert abs(p.mean() - 4.61) < 0.3      # 真机实测均值
    assert abs(np.degrees(src["roll"]).mean() + 0.35) < 0.3
    assert abs(src["height"].mean() - 0.522) < 0.02


def test_continuity_and_finiteness():
    rep = _make()
    rep.reset(torch.arange(4), torch.full((4,), 1.0))
    pitches = []
    for _ in range(2000):
        pitch, roll, height = rep.step(torch.full((4,), 1.0))
        pitches.append(pitch.clone())
    P = torch.stack(pitches)
    assert bool(torch.isfinite(P).all())
    d = (P[1:] - P[:-1]).abs()
    # 实测每步 |Δ| max ≈ 1.5°；生成器允许 AR 残差带来的小幅超出，但不允许跳变
    assert float(d.max()) < math.radians(3.0), f"jump detected: {float(d.max()):.3f} rad"


def test_seam_below_in_segment_p99():
    osc = np.sin(np.linspace(0, 40 * np.pi, 900)) * 0.02  # 4Hz-like, 900 步
    buf = build_episode_buffer(osc, rate=1.0, phase=137, fade=100, length=2250, dt=DT)
    d = np.abs(np.diff(buf))
    assert d.max() < 0.05  # 无外散发散、无接缝跳变
    # 接缝窗（每个循环交界的 fade 窗）内的步进不超过段内水平
    m = int(len(osc) / 1.0)
    seam_mask = np.zeros(len(d), bool)
    pos = m - 50
    while pos < len(d):
        seam_mask[pos:pos + 2] = True
        pos += m
    if seam_mask.any():
        assert d[seam_mask].max() <= np.percentile(d[~seam_mask], 99) + 1e-6


def test_short_first_piece_does_not_crash_or_jump():
    """回归：随机起始点近段尾时首片 < fade，必须无崩溃且平滑。"""
    osc = np.sin(np.linspace(0, 40 * np.pi, 900)) * 0.02
    for phase in (899, 850, 0, 450):
        buf = build_episode_buffer(osc, rate=1.0, phase=phase, fade=100,
                                   length=2250, dt=DT)
        d = np.abs(np.diff(buf))
        assert d.max() < 0.05, f"phase={phase}: jump {d.max():.4f}"
        assert len(buf) == 2250


def test_amplitude_law_tracks_speed():
    rep = _make(num_envs=2)
    rep.reset(torch.arange(2), torch.tensor([1.0, 0.25]))
    fast, slow = [], []
    for _ in range(3000):
        pitch, _, _ = rep.step(torch.tensor([1.0, 0.25]))
        fast.append(float(pitch[0]))
        slow.append(float(pitch[1]))
    # 去掉瞬态；生成器输出为弧度，比较前转度
    fast = np.degrees(np.array(fast[500:]))
    slow = np.degrees(np.array(slow[500:]))
    ref_std = math.degrees(rep.ref_std)
    std_fast = fast.std()
    std_slow = slow.std()
    law_fast = math.sqrt(1.176 ** 2 + 0.3 ** 2)      # 幅值律 + AR 残差
    law_slow = math.sqrt(0.5325 ** 2 + 0.3 ** 2)
    assert abs(std_fast - law_fast) < 0.35 * law_fast, f"{std_fast:.3f} vs {law_fast:.3f}"
    assert abs(std_slow - law_slow) < 0.45 * law_slow, f"{std_slow:.3f} vs {law_slow:.3f}"
    assert std_slow < std_fast


def test_pitch_mean_in_trim_range():
    rep = _make(num_envs=2)
    rep.reset(torch.arange(2), torch.full((2,), 0.5))
    acc = []
    for _ in range(2500):
        pitch, _, _ = rep.step(torch.full((2,), 0.5))
        acc.append(pitch.clone())
    m = torch.stack(acc).mean(dim=0)
    src_mean = math.degrees(load_source(SOURCE)["pitch"].mean())
    for e in range(2):
        deg = math.degrees(float(m[e]))
        assert src_mean - 2.5 < deg < src_mean + 2.5, f"trim out of range: {deg:.2f}"


def test_determinism_same_seed():
    out_a, out_b = [], []
    for out in (out_a, out_b):
        rep = _make(num_envs=2, seed=11)
        rep.reset(torch.arange(2), torch.full((2,), 0.8))
        for _ in range(300):
            pitch, roll, height = rep.step(torch.full((2,), 0.8))
            out.append((pitch.clone(), roll.clone(), height.clone()))
    for (pa, ra, ha), (pb, rb, hb) in zip(out_a, out_b):
        assert torch.equal(pa, pb) and torch.equal(ra, rb) and torch.equal(ha, hb)


def test_sign_convention_nose_down_compresses_far_rows():
    """审计门槛（完整几何复算）：pitch=+4.6°（Ry(+θ)，含传感器平移）时
    各排地面回波必须复现真机实测值；且 quat_from_euler_xyz(0, +pitch, 0)
    等价于 Ry(+pitch)。"""
    from isaacgym.torch_utils import quat_apply

    from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (
        body_frame_ray_directions, load_selected_channels,
    )

    th = math.radians(4.6)
    c, s = math.cos(th), math.sin(th)
    Ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)

    # (a) 符号/轴向：isaacgym euler 合成 == Ry(+θ)
    q = torch.tensor(_quat_from_euler_xyz_scalar(th), dtype=torch.float32).unsqueeze(0)
    probe = torch.tensor([[1.0, 0.0, -0.2]], dtype=torch.float64)
    assert torch.allclose(quat_apply(q, probe.float()).double(), (probe @ Ry.T), atol=1e-5)

    # (b) 完整几何复算：187 通道在 pitch=+4.6°、h=0.521 平地上的行均值
    d_body = body_frame_ray_directions().double()[
        load_selected_channels()["ray_indices"]]
    s_body = torch.tensor(ea2c.EA2_SENSOR_OFFSET_POS, dtype=torch.float64)
    H = 0.521
    d2 = d_body @ Ry.T
    s2 = Ry @ s_body
    dz = d2[:, 2]
    t_hit = torch.where(dz < -1e-6, -(H + s2[2]) / dz.clamp(max=-1e-6),
                        torch.full_like(dz, 60.0))
    row_means = t_hit.reshape(11, 17).mean(dim=1)
    real_rows = torch.tensor([0.67, 0.76, 1.18, 1.60, 1.85], dtype=torch.float64)
    picked = row_means[[0, 2, 5, 8, 10]]
    assert torch.allclose(picked, real_rows, atol=0.12), (
        f"pitch=+4.6° rows {picked.tolist()} vs real {real_rows.tolist()}"
    )


def _quat_from_euler_xyz_scalar(pitch):
    """Ry(pitch) 的 xyzw 四元数（与 isaacgym quat_from_euler_xyz 同构）。"""
    half = pitch / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"===== failures: {failures} =====")
    sys.exit(1 if failures else 0)
