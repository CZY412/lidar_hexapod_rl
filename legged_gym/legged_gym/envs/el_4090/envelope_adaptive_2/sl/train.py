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

# Frozen envelope contract bounds (pinned by tests/ea2/test_contracts.py and
# identical to the tensors hardcoded in sl/evaluate.py -- do not duplicate the
# spec-loading chain here, it pulls isaacgym through legged_gym.utils).
_LOW = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9])
_HIGH = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6])

# Collision semantics of the env reward (cfg.envelope.margin/soft_margin) and
# the env's floor-pinned definition (oracle_floor_pinned metric).
_SAFE_MARGIN = 0.10
_SAFE_SOFT = 0.10
_FLOOR_PINNED_VIOLATION = 0.05


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


def aux_memory_loss(
    pred: torch.Tensor, y: torch.Tensor, k: int, mode: str
) -> torch.Tensor:
    """Auxiliary memory loss for one probe horizon ``k``.

    ``pred`` is the probe's readout of the GRU hidden sequence ``h_t`` with the
    same ``(L, B, 5)`` layout as the target sequence ``y``.

    * ``mode="recall"``  -- ``pred[t]`` must equal ``y[t-k]``: reconstruct the
      envelope from ``k`` frames ago.  Pure memory demand: the past state
      cannot be derived from the current view, so the only path is retention
      in ``h``.
    * ``mode="forward"`` -- ``pred[t]`` must equal ``y[t+k]``: predict the
      envelope ``k`` frames ahead.  Mixed current-view extrapolation and
      retention; the gradient lands on ``h`` at the (last-sight) write moment.

    Frames without a valid partner are dropped (no padding, no wraparound).
    """
    L = y.shape[0]
    if not 0 < k < L:
        raise ValueError(f"aux horizon k={k} must be in (0, {L})")
    if mode == "recall":
        return torch.nn.functional.mse_loss(pred[k:], y[:-k])
    if mode == "forward":
        return torch.nn.functional.mse_loss(pred[: L - k], y[k:])
    raise ValueError(f"unknown aux_mode: {mode!r}")


def s_to_params(s: torch.Tensor) -> torch.Tensor:
    """Normalised extents -> envelope params (the deployment mapping)."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
        _physical_min_max,
    )

    min_v, max_v = _physical_min_max(_LOW, _HIGH)
    return min_v.to(s.device) + s * (max_v - min_v).to(s.device)


def batch_safety_loss(
    s_pred: torch.Tensor,
    heading: torch.Tensor,
    pos: torch.Tensor,
    map_ids: torch.Tensor,
    dfs: torch.Tensor,
):
    """Differentiable collision loss for a batch of frames.

    Uses *bilinear* distance-field sampling: the env-reward lookup is nearest
    neighbour, i.e. piecewise-constant in the sample position, so its gradient
    w.r.t. the envelope parameters is exactly zero -- unusable as an SL loss.
    The bilinear readout is piecewise-linear and its gradient points down the
    local field slope (toward the obstacle).

    Frames of different maps are evaluated group by group: a violation frame
    can only be indexed against its *own* map's field.  Floor-pinned frames
    (the minimum envelope already violates; monitored by the env as
    ``oracle_floor_pinned``) are excluded from the loss -- there the gradient
    would be a constant collapse pressure with no escape direction.

    Returns ``(mean_violation, trigger_rate)`` over the 24 boundary samples of
    the non-floor-pinned frames (or zeros when every frame is floor-pinned).
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
        _hex_sample_violations,
    )

    params = s_to_params(s_pred.clamp(0.0, 1.0))
    losses, weights, triggers = [], [], []
    for m in map_ids.unique():
        sel = map_ids == m
        n = int(sel.sum())
        with torch.no_grad():  # mask only; must not carry gradient
            min_params = s_to_params(torch.zeros_like(s_pred[sel]))
            floor = (
                _hex_sample_violations(
                    min_params, heading[sel], pos[sel], dfs[m],
                    margin=_SAFE_MARGIN, soft_margin=_SAFE_SOFT,
                    sampling="bilinear",
                ).max(dim=-1).values
                > _FLOOR_PINNED_VIOLATION
            )
        v = _hex_sample_violations(
            params[sel], heading[sel], pos[sel], dfs[m],
            margin=_SAFE_MARGIN, soft_margin=_SAFE_SOFT,
            sampling="bilinear",
        )[..., :24]
        frame_loss = v.mean(dim=-1)
        keep = ~floor
        n_keep = int(keep.sum())
        if n_keep:
            losses.append(frame_loss[keep].sum())
            weights.append(n_keep)
            triggers.append(float((frame_loss > 0)[keep].float().sum()) / n_keep)
    if not losses:
        zero = params.sum() * 0.0
        return zero, 0.0
    total = torch.stack(losses).sum() / sum(weights)
    rate = sum(t * w for t, w in zip(triggers, weights)) / sum(weights)
    return total, rate


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

    # per-map distance fields on device for the differentiable safety loss
    safe_lambda = float(getattr(cfg.train, "safe_lambda", 0.0))
    dfs = None
    if safe_lambda > 0:
        dfs = torch.stack(
            [torch.as_tensor(m.distance_field, dtype=torch.float32) for m in dataset.maps]
        ).to(dev)

    best_r2, best_state, best_epoch = -np.inf, None, -1
    epochs_without_gain = 0
    stopped_early = False
    t0 = time.time()

    for epoch in range(cfg.train.epochs):
        net.train()
        perm = torch.randperm(len(tr))
        running = 0.0
        running_aux = 0.0
        running_safe, safe_frames = 0.0, 0
        aux_beta = float(getattr(cfg.train, "aux_beta", 0.5))
        for i in range(0, len(tr), bs):
            b = tr[perm[i : i + bs]]
            x = dataset.batch(b).transpose(0, 1).to(dev)
            y = tgt[b].transpose(0, 1).to(dev)
            use_aux = net.aux_heads and aux_beta > 0
            if use_aux or safe_lambda > 0:
                main_out, h_seq = net.forward_with_aux(x)
            else:
                main_out = net(x)
            main_mse = torch.nn.functional.mse_loss(main_out, y)
            loss = main_mse
            running += float(main_mse) * len(b)
            if use_aux:
                for k_str, head in net.aux_heads.items():
                    aux_l = aux_memory_loss(head(h_seq), y, int(k_str), cfg.model.aux_mode)
                    loss = loss + aux_beta * aux_l
                    running_aux += float(aux_l) * len(b)
            if safe_lambda > 0:
                # flatten the (L, B, ...) window batch into frames: the loss
                # contract is a flat (N, 5) frame batch with per-frame fields
                L_frames = main_out.shape[0]
                safe_l, trig = batch_safety_loss(
                    main_out.reshape(-1, 5),
                    dataset.heading[b].transpose(0, 1).reshape(-1).to(dev),
                    dataset.pos[b].transpose(0, 1).reshape(-1, 2).to(dev),
                    dataset.map_id[b].repeat_interleave(L_frames).to(dev),
                    dfs,
                )
                loss = loss + safe_lambda * safe_l
                running_safe += float(safe_l) * len(b)
                safe_frames += len(b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
            opt.step()

        train_mse = running / len(tr)
        aux_mse_epoch = running_aux / len(tr) if use_aux else None
        safe_epoch = running_safe / max(1, safe_frames) if safe_lambda > 0 else None
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
            astr = (
                f"  aux_mse({cfg.model.aux_mode})={aux_mse_epoch:.5f}"
                if aux_mse_epoch is not None
                else ""
            )
            sstr = (
                f"  safe_violation={safe_epoch:.5f}"
                if safe_epoch is not None
                else ""
            )
            print(
                f"  [train] ep{epoch + 1:>3}  train_mse={train_mse:.5f}  {vstr}"
                f"{astr}{sstr}  ({time.time() - t0:.0f}s)"
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

    # per-k aux loss on the val split: how well h decodes the (recalled or
    # future) envelope -- the direct memory-utilisation readout
    aux_val_mse = None
    if net.aux_heads:
        aux_val_mse = {}
        with torch.no_grad():
            for k_str, head in net.aux_heads.items():
                errs = []
                for i in range(0, len(va), 128):
                    x = dataset.batch(va[i : i + 128]).transpose(0, 1).to(dev)
                    y = tgt[va[i : i + 128]].transpose(0, 1).to(dev)
                    _, h_seq = net.forward_with_aux(x)
                    errs.append(float(aux_memory_loss(head(h_seq), y, int(k_str), cfg.model.aux_mode)))
                aux_val_mse[f"k{int(k_str)}"] = float(np.mean(errs))

    # val-set safety metrics: trigger rate (frames whose prediction violates)
    # and mean violation -- the collapse guard for the safety loss
    safe_val = None
    if safe_lambda > 0:
        trig, viol, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va), 128):
                idx = va[i : i + 128]
                x = dataset.batch(idx).transpose(0, 1).to(dev)
                main_out = net(x)
                L_frames = main_out.shape[0]
                l, t = batch_safety_loss(
                    main_out.reshape(-1, 5),
                    dataset.heading[idx].transpose(0, 1).reshape(-1).to(dev),
                    dataset.pos[idx].transpose(0, 1).reshape(-1, 2).to(dev),
                    dataset.map_id[idx].repeat_interleave(L_frames).to(dev),
                    dfs,
                )
                b_n = len(idx)
                viol += float(l) * b_n
                trig += float(t) * b_n
                n += b_n
        safe_val = {"trigger_rate": trig / n, "mean_violation": viol / n}

    return {
        "net": net,
        "history": hist,
        "best_epoch": best_epoch,
        "best_val_r2": float(best_r2),
        "val": final_val,
        "train": final_train,
        "aux_val_mse": aux_val_mse,
        "safe_val": safe_val,
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
