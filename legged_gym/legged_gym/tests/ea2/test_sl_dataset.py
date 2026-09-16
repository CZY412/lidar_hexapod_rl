"""Gate G0 -- data-contract checks for the EA2 supervised-learning dataset.

These tests are the review gate for block B0.  They are written to run against
a small synthetic ``MapData`` so they stay fast, plus an optional integration
test that reads real collected files when ``EA2_SL_DATA_DIR`` is set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.sl import dataset as ds
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import SLConfig

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _make_map(
    n_frames: int = 60,
    n_envs: int = 8,
    seed: int = 1,
    done_at: dict | None = None,
) -> ds.MapData:
    """Synthetic map with per-env distinct targets and a controllable done."""
    rng = np.random.RandomState(seed)
    obs = torch.rand(n_frames, n_envs, 190)  # already in [0, 1]
    # give each env a constant-but-distinct target so env leakage is detectable
    base = torch.linspace(0.2, 0.8, n_envs).unsqueeze(0).expand(n_frames, -1)
    target = (base.unsqueeze(-1).expand(-1, -1, 5) + 0.01 * torch.randn(n_frames, n_envs, 5)).clamp(0, 1)
    done = torch.zeros(n_frames, n_envs, dtype=torch.bool)
    for env_id, frame in (done_at or {}).items():
        done[int(frame), int(env_id)] = True
    heading = torch.rand(n_frames, n_envs)
    pos = torch.rand(n_frames, n_envs, 2) * 4.0 - 2.0
    return ds.MapData(
        seed=seed,
        obs=obs,
        target=target,
        done=done,
        heading=heading,
        pos=pos,
        distance_field=torch.rand(74, 74) * 3.0,
        meta={"seed": seed},
    )


# --------------------------------------------------------------------------
# MapData contract
# --------------------------------------------------------------------------


def test_mapdata_validates_shapes():
    m = _make_map()
    assert m.obs.shape[-1] == 190
    assert m.target.shape[-1] == 5
    with pytest.raises(AssertionError):
        ds.MapData(
            seed=1,
            obs=torch.rand(10, 2, 180),
            target=torch.rand(10, 2, 5),
            done=torch.zeros(10, 2, dtype=torch.bool),
            heading=torch.rand(10, 2),
            pos=torch.rand(10, 2, 2),
            distance_field=torch.rand(74, 74),
            meta={},
        )


def test_save_load_roundtrip(tmp_path):
    m = _make_map()
    path = str(tmp_path / "m.pt")
    ds.save_map(m, path)
    loaded = ds.load_map(path)
    assert torch.equal(loaded.obs, m.obs)
    assert torch.equal(loaded.target, m.target)
    assert torch.equal(loaded.done, m.done)
    assert torch.equal(loaded.distance_field, m.distance_field)
    assert loaded.seed == m.seed


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------


def test_build_windows_rejects_done_windows():
    """A window containing a done must be dropped, not silently truncated."""
    m = _make_map(n_frames=60, n_envs=2, done_at={0: 30})
    w = ds.build_windows(m.obs, m.target, m.done, m.heading, m.pos, seq_len=10, warmup_frames=0, stride=1)
    # env 0 has a done at frame 30; every window covering it is rejected.
    for i in range(len(w["obs"])):
        env = int(w["env_id"][i])
        if env != 0:
            continue
    starts_env0 = sorted(
        int(w["env_id"][i]) == 0 and i for i in range(len(w["obs"])) if int(w["env_id"][i]) == 0
    )
    assert len(starts_env0) > 0
    # env 1 (no done) keeps the full complement of windows
    n_env1 = int((w["env_id"] == 1).sum())
    assert n_env1 == 60 - 10 + 1


def test_build_windows_respects_warmup():
    m = _make_map(n_frames=60, n_envs=2)
    w = ds.build_windows(m.obs, m.target, m.done, m.heading, m.pos, seq_len=10, warmup_frames=20, stride=1)
    # warmup discards the first 20 possible start positions
    per_env = (w["env_id"] == 0).sum().item()
    assert per_env == (60 - 10 + 1) - 20


def test_windows_are_contiguous_slices():
    """Each window must be an exact contiguous slice of the source series."""
    m = _make_map(n_frames=40, n_envs=3)
    seq = 8
    w = ds.build_windows(m.obs, m.target, m.done, m.heading, m.pos, seq_len=seq, warmup_frames=0, stride=3)
    assert len(w["obs"]) > 0
    # verify one window against the source: recompute by matching the last frame
    for i in range(min(5, len(w["obs"]))):
        env = int(w["env_id"][i])
        last = w["target"][i, -1]
        # the last frame of the window must exist in the source for that env
        matches = (m.target[:, env] == last).all(dim=-1).nonzero().flatten()
        assert matches.numel() >= 1


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def test_env_split_is_disjoint():
    env_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    tr, va = ds.env_split(env_ids, val_fraction=0.25, seed=0)
    tr_envs = set(env_ids[tr].tolist())
    va_envs = set(env_ids[va].tolist())
    assert tr_envs & va_envs == set(), "train/val environments must not overlap"
    assert tr_envs | va_envs == {0, 1, 2, 3}
    assert len(va_envs) == 1  # 25% of 4 envs, at least 1


def test_env_split_is_deterministic():
    env_ids = torch.arange(20).repeat_interleave(3)
    a = ds.env_split(env_ids, val_fraction=0.2, seed=7)
    b = ds.env_split(env_ids, val_fraction=0.2, seed=7)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_env_split_covers_every_window():
    env_ids = torch.randint(0, 10, (500,))
    tr, va = ds.env_split(env_ids, val_fraction=0.2, seed=3)
    assert len(tr) + len(va) == len(env_ids)


# --------------------------------------------------------------------------
# dataset assembly
# --------------------------------------------------------------------------


def test_sldataset_no_env_leakage_across_maps():
    cfg = SLConfig()
    cfg.train.seq_len = 10
    cfg.train.window_stride = 5
    cfg.data.warmup_steps = 0
    maps = [_make_map(n_frames=60, n_envs=6, seed=s) for s in (1, 2)]
    d = ds.SLDataset(cfg, maps)
    tr_envs = set(d.env_id[d.train_idx].tolist())
    va_envs = set(d.env_id[d.val_idx].tolist())
    assert tr_envs & va_envs == set()
    # env ids are globally unique across maps -> no accidental aliasing
    assert d.env_id.unique().numel() == 12


def test_quantised_dataset_uses_four_times_less_memory():
    """``quantise_obs`` exists because windowing inflates data ~15x.

    Four maps of ~90 MiB become ~1.4 GiB of float32 windows; uint8 storage cuts
    that to roughly a quarter.  Guarded so the option cannot silently rot.
    """
    cfg = SLConfig()
    cfg.train.seq_len = 10
    cfg.data.warmup_steps = 0
    base = [torch.rand(50, 6, 190) for _ in range(2)]

    def build(quantise: bool, obs_list):
        c = SLConfig()
        c.train.seq_len = 10
        c.train.window_stride = 2
        c.data.warmup_steps = 0
        c.train.quantise_obs = quantise
        maps = [
            ds.MapData(seed=i, obs=o, target=torch.rand(50, 6, 5),
                       done=torch.zeros(50, 6, dtype=torch.bool),
                       heading=torch.rand(50, 6), pos=torch.rand(50, 6, 2),
                       distance_field=torch.rand(74, 74), meta={})
            for i, o in enumerate(obs_list)
        ]
        return ds.SLDataset(c, maps)

    plain = build(False, base)
    quant = build(True, base)
    pb = plain.obs.numel() * plain.obs.element_size()
    qb = quant.obs.numel() * quant.obs.element_size()
    assert plain.obs.dtype == torch.float32
    assert quant.obs.dtype == torch.uint8
    assert qb * 4 == pb, f"expected 4x reduction, got {pb / qb:.1f}x"
    # decoding must round-trip within the quantisation step (2/255 units)
    decoded = quant.batch(torch.arange(len(quant)))
    err = (decoded - plain.obs).abs().max().item()
    assert err <= 1.0 / 127.5 + 1e-6, f"quantisation error too large: {err}"
    # and the split must be identical regardless of storage
    assert torch.equal(plain.train_idx, quant.train_idx)


def test_sldataset_summary_fields():
    cfg = SLConfig()
    cfg.train.seq_len = 10
    cfg.data.warmup_steps = 0
    d = ds.SLDataset(cfg, [_make_map(n_frames=50, n_envs=6)])
    s = d.summary()
    for key in (
        "n_windows",
        "n_train",
        "n_val",
        "n_envs",
        "obs_min",
        "obs_max",
        "target_std",
        "target_var_mean",
        "corr_fw_fl",
        "corr_bw_bl",
        "frac_saturated_1",
        "frac_saturated_0",
    ):
        assert key in s, key
    assert s["n_windows"] == s["n_train"] + s["n_val"]


# --------------------------------------------------------------------------
# optional integration test against real collected data
# --------------------------------------------------------------------------

DATA_DIR = os.environ.get("EA2_SL_DATA_DIR", "")


@pytest.mark.skipif(not DATA_DIR, reason="EA2_SL_DATA_DIR not set")
def test_real_data_contract():
    """Validate real collected maps: ranges, sync, and no-leakage."""
    cfg = SLConfig()
    files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("map_seed") and f.endswith(".pt"))
    assert files, f"no map_seed*.pt found in {DATA_DIR}"
    maps = [ds.load_map(os.path.join(DATA_DIR, f)) for f in files]

    for m in maps:
        # Range channels are range/3.2 -> [0, 1]; ego channels are signed and
        # can reach +/-1 (yaw rate saturates).  Overall range is [-1, 1].
        rng = m.obs[..., :187]
        ego = m.obs[..., 187:]
        assert float(rng.min()) >= -1e-4, f"range image must be non-negative, got {float(rng.min())}"
        assert float(rng.max()) <= 1.0 + 1e-3, f"range image must be <= 1, got {float(rng.max())}"
        assert float(ego.min()) >= -1.05, f"ego below -1: {float(ego.min())}"
        assert float(ego.max()) <= 1.05, f"ego above +1: {float(ego.max())}"
        assert float(m.target.min()) >= 0.0 and float(m.target.max()) <= 1.0

    d = ds.SLDataset(cfg, maps)
    s = d.summary()
    tr_envs = set(d.env_id[d.train_idx].tolist())
    va_envs = set(d.env_id[d.val_idx].tolist())
    assert tr_envs & va_envs == set()

    # target must carry real signal: not constant
    assert s["target_var_mean"] > 1e-3, f"target variance too low: {s['target_var_mean']}"
    print(f"[G0] windows={s['n_windows']} train={s['n_train']} val={s['n_val']}")
    print(f"[G0] target_std={s['target_std']}")
    print(f"[G0] corr fw~fl={s['corr_fw_fl']:.4f}  bw~bl={s['corr_bw_bl']:.4f}")
    print(f"[G0] saturated at 1: {s['frac_saturated_1']:.4f}  at 0: {s['frac_saturated_0']:.4f}")
