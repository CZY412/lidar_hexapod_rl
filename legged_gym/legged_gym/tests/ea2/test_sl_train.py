"""Gate G2 -- training-behaviour checks for the EA2 supervised pipeline.

These tests use a small synthetic dataset so they run in seconds.  The
*thresholds* for the real run are asserted from the metrics JSON produced by
``sl/scripts/train.py`` when ``EA2_SL_METRICS`` points at it, so the numbers
that actually matter are still gated.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.sl import dataset as ds
from legged_gym.envs.el_4090.envelope_adaptive_2.sl import train as slt
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import SLConfig

METRICS = os.environ.get("EA2_SL_METRICS", "")


def _cfg(seq_len: int = 10, epochs: int = 6, warmup: int = 0) -> SLConfig:
    cfg = SLConfig()
    cfg.train.seq_len = seq_len
    cfg.train.epochs = epochs
    cfg.train.batch_size = 16
    cfg.train.window_stride = 3
    cfg.data.warmup_steps = warmup
    return cfg


def _dataset(cfg: SLConfig, n_frames: int = 40, n_envs: int = 8) -> ds.SLDataset:
    torch.manual_seed(0)
    obs = torch.rand(n_frames, n_envs, 190)
    # make the target a smooth function of a few observation channels so that
    # it is learnable, plus a small per-env bias
    w = torch.randn(190, 5) * 0.05
    lin = obs.reshape(-1, 190) @ w
    noise = 0.3 * torch.randn(n_frames * n_envs, 1)
    target = torch.sigmoid(lin + noise).reshape(n_frames, n_envs, 5)
    done = torch.zeros(n_frames, n_envs, dtype=torch.bool)
    heading = torch.rand(n_frames, n_envs)
    pos = torch.rand(n_frames, n_envs, 2)
    m = ds.MapData(
        seed=1, obs=obs, target=target, done=done, heading=heading, pos=pos,
        distance_field=torch.rand(74, 74), meta={},
    )
    return ds.SLDataset(cfg, [m])


def test_train_reduces_loss():
    cfg = _cfg(epochs=8)
    d = _dataset(cfg)
    res = slt.train(cfg, d, device="cpu", verbose=False)
    assert res["history"].val_mse[-1] < res["history"].val_mse[0]
    assert res["history"].train_mse[-1] < res["history"].train_mse[0]


def test_train_returns_expected_fields():
    cfg = _cfg(epochs=3)
    d = _dataset(cfg)
    res = slt.train(cfg, d, device="cpu", verbose=False)
    for key in ("history", "best_epoch", "best_val_r2", "val", "train", "n_train", "n_val"):
        assert key in res
    assert 1 <= res["best_epoch"] <= cfg.train.epochs
    assert res["n_train"] + res["n_val"] == len(d)


def test_best_epoch_restores_best_weights():
    """The returned net must correspond to the best validation epoch."""
    cfg = _cfg(epochs=10)
    d = _dataset(cfg)
    res = slt.train(cfg, d, device="cpu", verbose=False)
    # recomputing val r2 with the returned net must match best_val_r2
    pred = slt._predict(res["net"], d, d.val_idx, "cpu")
    m = slt.compute_metrics(pred, d.target[d.val_idx])
    assert abs(m["r2"] - res["best_val_r2"]) < 1e-4


def test_metrics_shapes_and_keys():
    pred = torch.rand(4, 10, 5)
    tgt = torch.rand(4, 10, 5)
    m = slt.compute_metrics(pred, tgt)
    assert len(m["pc_var"]) == 5 and len(m["pc_r2"]) == 5
    assert abs(sum(m["pc_var"]) - 1.0) < 1e-6, "pc_var must be a distribution"
    assert set(m["mse_per_dim"].keys()) == set(m["r2_per_dim"].keys())
    assert set(m["subspace_r2"].keys()) == {"top1", "top2", "top3"}


def test_metrics_perfect_prediction_gives_r2_one():
    tgt = torch.rand(4, 10, 5)
    m = slt.compute_metrics(tgt.clone(), tgt)
    assert abs(m["r2"] - 1.0) < 1e-5
    for v in m["r2_per_dim"].values():
        assert abs(v - 1.0) < 1e-5


def test_pca_reports_top1_dominance_for_rank_deficient_target():
    """A nearly rank-1 target must show up as a dominant PC0."""
    base = torch.rand(8, 6, 1)
    tgt = (base.expand(-1, -1, 5) + 0.001 * torch.randn(8, 6, 5)).clamp(0, 1)
    m = slt.compute_metrics(tgt.clone(), tgt)
    assert m["pc_var"][0] > 0.99


def test_checkpoint_roundtrip(tmp_path):
    cfg = _cfg(epochs=2)
    d = _dataset(cfg)
    res = slt.train(cfg, d, device="cpu", verbose=False)
    path = str(tmp_path / "ck.pt")
    slt.save_checkpoint(res, cfg, path)
    assert os.path.exists(path)
    assert os.path.exists(os.path.splitext(path)[0] + "_metrics.json")
    from legged_gym.envs.el_4090.envelope_adaptive_2.sl.evaluate import load_checkpoint

    net2, meta = load_checkpoint(path)
    assert torch.allclose(net2.state_dict()["mlp.4.weight"], res["net"].state_dict()["mlp.4.weight"])


# --------------------------------------------------------------------------
# thresholds against the real training run
# --------------------------------------------------------------------------


@pytest.mark.skipif(not METRICS, reason="EA2_SL_METRICS not set")
def test_real_run_meets_quality_gates():
    """Assert the real training run clears the G2 thresholds."""
    with open(METRICS) as f:
        m = json.load(f)
    val = m["val"]
    print(f"[G2] best_epoch={m['best_epoch']} val_r2={val['r2']:.4f} mse={val['mse']:.5f}")
    print(f"[G2] r2_per_dim={ {k: round(v, 3) for k, v in val['r2_per_dim'].items()} }")
    print(f"[G2] mse_per_dim={ {k: round(v, 5) for k, v in val['mse_per_dim'].items()} }")

    assert val["r2"] >= 0.70, f"val R2 too low: {val['r2']:.4f}"
    # train/val gap: mild overfitting is expected, runaway overfitting is not
    gap = m["train"]["r2"] - val["r2"]
    assert gap <= 0.15, f"train/val gap too large: {gap:.4f}"
    # best epoch must be inside the run, and not stuck at 1 (which would mean
    # the very first epoch was already the peak -> something is wrong)
    assert m["best_epoch"] >= 2, "best epoch at 1 -- training did not improve"
    # PC0 (overall envelope size) must be learned well
    assert val["pc_r2"][0] >= 0.75, f"PC0 R2 too low: {val['pc_r2'][0]:.3f}"
    # no dimension may be actively harmful (R2 strongly negative)
    worst = min(val["r2_per_dim"].values())
    assert worst > -0.25, f"a dimension is badly wrong, worst R2={worst:.3f}"
    # the envelope must actually move
    ratio = val["temporal_std"] / max(val["temporal_std_oracle"], 1e-8)
    print(f"[G2] temporal_std_ratio={ratio:.3f}")
    assert ratio >= 0.70, f"envelope too static: ratio {ratio:.3f}"
