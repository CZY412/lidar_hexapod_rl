"""Training loop and offline metrics for the EA2 supervised pipeline (B2/B3).

Why plain MSE and not PPO
-------------------------
The environment reward is ``r = -3 * MSE(a, oracle)``: it is immediate,
deterministic, and has no cross-step credit assignment.  Training on the oracle
directly is therefore *strictly more informative* than sampling that same
signal through PPO, and it costs seconds instead of hours.

Reported metrics
----------------
``r2``            overall coefficient of determination
``r2_per_dim``    per-parameter R2
``mse_per_dim``   per-parameter MSE in normalised units
``pca_*``         R2 broken down over the target's principal components

The PCA view matters because the oracle target is close to low rank: a high
overall R2 can be carried almost entirely by "overall envelope size" (PC0)
while the dimension-specific detail is barely learned.  Reporting both keeps
that failure mode visible.

A note on R2 comparability
--------------------------
R2 depends on target variance, which changed materially when ``oracle_margin``
moved from 0.10 to 0.20 (variance ~0.043 -> ~0.082, and the front/limit
correlation dropped from 0.77 to 0.49).  R2 numbers are therefore only
comparable within one oracle configuration -- always read ``mse_per_dim`` too.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .dataset import SLDataset
from .model import EnvelopeNet
from .sl_config import PARAM_NAMES, SLConfig


@dataclass
class History:
    train_mse: List[float] = field(default_factory=list)
    val_mse: List[float] = field(default_factory=list)
    val_r2: List[float] = field(default_factory=list)

    def best_epoch(self) -> int:
        return int(np.argmax(self.val_r2)) if self.val_r2 else -1


def _predict(
    net: nn.Module, dataset, idx: torch.Tensor, device: str, batch: int = 128
):
    """Run the net over ``dataset`` windows ``idx``.

    Takes the *dataset* rather than a raw tensor so that quantised storage
    (``SLDataset.quantised``) is decoded transparently.
    """
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            chunk = dataset.batch(idx[i : i + batch]).transpose(0, 1).to(device)
            preds.append(net(chunk).transpose(0, 1).cpu())
    return torch.cat(preds, dim=0).clamp(0.0, 1.0)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict:
    """Offline metrics for normalised-envelope predictions."""
    err = (pred - target) ** 2
    mse_per_dim = err.mean(dim=(0, 1))
    var_per_dim = target.var(dim=(0, 1))
    r2_per_dim = 1.0 - mse_per_dim / var_per_dim.clamp_min(1e-8)

    mse = float(err.mean())
    var = float(var_per_dim.mean())
    r2 = 1.0 - mse / max(var, 1e-8)

    # --- PCA decomposition of the target, then R2 along each component ---
    flat_t = target.reshape(-1, 5).numpy()
    mu = flat_t.mean(0)
    centred = flat_t - mu
    _, s, vt = np.linalg.svd(np.cov(centred.T), hermitian=True)
    ev = (s**2) / (s**2).sum()
    flat_p = pred.reshape(-1, 5).numpy()

    pc_var, pc_r2 = [], []
    for j in range(5):
        v = vt[j]
        yt = centred @ v
        pt = (flat_p - mu) @ v
        denom = float(yt.var())
        pc_var.append(float(ev[j]))
        pc_r2.append(float(1.0 - ((pt - yt) ** 2).mean() / denom) if denom > 1e-10 else float("nan"))

    # subspace R2 for the top-k components
    sub_r2 = {}
    for k in (1, 2, 3):
        P = vt[:k]
        yt = centred @ P.T
        pt = (flat_p - mu) @ P.T
        denom = float(yt.var(0).sum()) * len(yt)
        sub_r2[f"top{k}"] = float(1.0 - ((pt - yt) ** 2).sum() / denom) if denom > 1e-10 else float("nan")

    return {
        "mse": mse,
        "r2": r2,
        "target_var": var,
        "mse_per_dim": {n: float(x) for n, x in zip(PARAM_NAMES, mse_per_dim)},
        "r2_per_dim": {n: float(x) for n, x in zip(PARAM_NAMES, r2_per_dim)},
        "target_std_per_dim": {n: float(x) for n, x in zip(PARAM_NAMES, target.std(dim=(0, 1)))},
        "pc_var": pc_var,
        "pc_r2": pc_r2,
        "subspace_r2": sub_r2,
        # dynamic-ness: std over time, compared against oracle's own
        "temporal_std": float(pred.std(dim=1).mean()),
        "temporal_std_oracle": float(target.std(dim=1).mean()),
    }


def train(
    cfg: SLConfig,
    dataset: SLDataset,
    device: str = "cuda",
    log_every: int = 5,
    verbose: bool = True,
    patience: int = 0,
    val_every: int = 1,
) -> Dict:
    """Train ``EnvelopeNet``; returns metrics plus the best-epoch weights.

    Args:
        patience: stop after this many *validated* epochs without improvement
            (0 disables early stopping).  Validation improvements are only
            observable on epochs where validation actually runs.
        val_every: run validation every N epochs.  Validation is a full forward
            pass over the validation split and is easily the most expensive part
            of an epoch, so raising this is a cheap speed-up for large datasets.
            The final epoch is always validated.
    """
    dev = device if torch.cuda.is_available() else "cpu"
    net = EnvelopeNet(cfg.model).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.train.learning_rate)
    hist = History()

    tr, va = dataset.train_idx, dataset.val_idx
    tgt = dataset.target
    bs = cfg.train.batch_size

    best_r2, best_state, best_epoch = -np.inf, None, -1
    epochs_without_gain = 0
    stopped_early = False
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        net.train()
        perm = torch.randperm(len(tr))
        running = 0.0
        for i in range(0, len(tr), bs):
            b = tr[perm[i : i + bs]]
            x = dataset.batch(b).transpose(0, 1).to(dev)
            y = tgt[b].transpose(0, 1).to(dev)
            loss = torch.nn.functional.mse_loss(net(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
            opt.step()
            running += float(loss) * len(b)

        train_mse = running / len(tr)
        is_last = epoch == cfg.train.epochs - 1
        do_val = is_last or ((epoch + 1) % val_every == 0)
        val_metrics = None

        if do_val:
            val_pred = _predict(net, dataset, va, dev)
            val_metrics = compute_metrics(val_pred, tgt[va])
            hist.train_mse.append(train_mse)
            hist.val_mse.append(val_metrics["mse"])
            hist.val_r2.append(val_metrics["r2"])

            if val_metrics["r2"] > best_r2:
                best_r2 = val_metrics["r2"]
                best_epoch = epoch + 1
                epochs_without_gain = 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                epochs_without_gain += 1

            if patience and epochs_without_gain >= patience and not is_last:
                stopped_early = True

        if verbose and (epoch % log_every == 0 or is_last or stopped_early):
            vstr = (
                f"val_r2={val_metrics['r2']:.4f}  val_mse={val_metrics['mse']:.5f}"
                if val_metrics
                else "val skipped"
            )
            print(
                f"  [train] ep{epoch + 1:>3}  train_mse={train_mse:.5f}  {vstr}  "
                f"({time.time() - t0:.0f}s)"
            )
        if stopped_early:
            if verbose:
                print(f"  [train] early stop at epoch {epoch + 1} (patience={patience})")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    final_val = compute_metrics(_predict(net, dataset, va, dev), tgt[va])
    final_train = compute_metrics(_predict(net, dataset, tr, dev), tgt[tr])

    return {
        "net": net,
        "history": hist,
        "best_epoch": best_epoch,
        "best_val_r2": float(best_r2),
        "val": final_val,
        "train": final_train,
        "elapsed_s": time.time() - t0,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "stopped_early": stopped_early,
        "epochs_run": epoch + 1 if cfg.train.epochs else 0,
    }


def save_checkpoint(result: Dict, cfg: SLConfig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": result["net"].state_dict(),
            "cfg_model": cfg.model.__dict__,
            "best_epoch": result["best_epoch"],
            "best_val_r2": result["best_val_r2"],
            "val": result["val"],
        },
        path,
    )
    with open(os.path.splitext(path)[0] + "_metrics.json", "w") as f:
        json.dump(
            {
                "best_epoch": result["best_epoch"],
                "best_val_r2": result["best_val_r2"],
                "val": result["val"],
                # train metrics are needed to compute the generalisation gap
                "train": {
                    "r2": result["train"]["r2"],
                    "mse": result["train"]["mse"],
                    "r2_per_dim": result["train"]["r2_per_dim"],
                },
                "n_train": result["n_train"],
                "n_val": result["n_val"],
                "elapsed_s": result["elapsed_s"],
            },
            f,
            indent=2,
        )
