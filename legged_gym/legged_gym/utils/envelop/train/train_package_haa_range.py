"""Train a geometry-labelled, drop-in replacement for ``haa_range.pt``.

Example (small smoke run)::

    python -m legged_gym.utils.envelop.train.train_package_haa_range \
      --samples 64 --fold-samples 512 --output /tmp/haa_range_geometry.pt

Use substantially more samples (at least 10k) and the default fold resolution
for a checkpoint intended for locomotion training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterable

import torch
from torch import Tensor

try:  # Package import when called from the simulator tree.
    from ..network.haa_swing_range import (
        HaaRangeConfig,
        HaaRangeNetwork,
        apply_env_morphology_priors,
        load_envelope_condition_spec,
    )
    from .package_haa_range import (
        PackageHaaRangeTeacher,
        PackageHaaTeacherConfig,
        condition_footprint_support,
    )
except ImportError:  # Direct-file execution: avoid legged_gym.utils/__init__.py.
    from _standalone import load_haa_range_module
    from package_haa_range import (
        PackageHaaRangeTeacher,
        PackageHaaTeacherConfig,
        condition_footprint_support,
    )
    _haa_module = load_haa_range_module()
    HaaRangeConfig = _haa_module.HaaRangeConfig
    HaaRangeNetwork = _haa_module.HaaRangeNetwork
    apply_env_morphology_priors = _haa_module.apply_env_morphology_priors
    load_envelope_condition_spec = _haa_module.load_envelope_condition_spec


def make_dataset(
    teacher: PackageHaaRangeTeacher,
    condition_low: Tensor,
    condition_high: Tensor,
    *,
    samples: int,
    condition_transform,
    progress_every: int,
) -> tuple[Tensor, Tensor, int]:
    """Create valid labels; invalid geometric envelopes are explicitly dropped."""
    raw = condition_low.cpu() + torch.rand(samples, condition_low.numel()) * (condition_high.cpu() - condition_low.cpu())
    condition = condition_transform(raw)
    support = condition_footprint_support(
        condition, teacher.directions.cpu(), teacher.range_config.condition_names,
        body_margin=teacher.config.synthetic_body_margin,
    )
    kept_condition, kept_ranges = [], []
    rejected = 0
    started = time.monotonic()
    total = condition.shape[0]
    print(f"[label] start rows={total} sweep_steps={teacher.config.sweep_steps} "
          f"fold_samples={teacher.config.fold_samples} device={teacher.device}", flush=True)
    for index, (row, allowed) in enumerate(zip(condition, support), start=1):
        try:
            kept_ranges.append(teacher.label(row, allowed).cpu())
            kept_condition.append(row)
        except RuntimeError:
            rejected += 1
        if index % progress_every == 0 or index == total:
            elapsed = time.monotonic() - started
            rate = index / max(elapsed, 1e-9)
            eta = (total - index) / max(rate, 1e-9)
            print(f"[label] {index}/{total} valid={len(kept_condition)} dropped={rejected} "
                  f"rate={rate:.2f} rows/s eta={eta / 60.0:.1f} min", flush=True)
    if not kept_condition:
        raise RuntimeError("No valid labels. Increase synthetic_body_margin or provide planner support data.")
    return torch.stack(kept_condition), torch.stack(kept_ranges), rejected


def make_dataset_from_support(
    teacher: PackageHaaRangeTeacher,
    path: Path,
    *,
    condition_transform,
    progress_every: int,
    max_rows: int | None,
) -> tuple[Tensor, Tensor, int]:
    """Label recorded planner envelopes.

    ``path`` is a Torch dictionary with ``condition`` shaped ``[N,8]`` and
    ``allowed_support`` shaped ``[N,K]`` (``K=--directions``, default 32).
    This is the preferred route for asymmetric LiDAR envelopes: the teacher
    sees every support face even though the legacy student only sees its 8-D
    policy-compatible projection.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or {"condition", "allowed_support"}.difference(payload):
        raise ValueError("--support-dataset must contain tensors 'condition' and 'allowed_support'")
    condition = torch.as_tensor(payload["condition"], dtype=torch.float32)
    support = torch.as_tensor(payload["allowed_support"], dtype=torch.float32)
    if condition.ndim != 2 or condition.shape[1] != len(teacher.range_config.condition_names):
        raise ValueError("dataset condition must have shape [N, 8]")
    if support.shape != (condition.shape[0], teacher.config.directions):
        raise ValueError(f"dataset allowed_support must have shape [N, {teacher.config.directions}]")
    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("--max-label-rows must be positive")
        condition, support = condition[:max_rows], support[:max_rows]
    condition = condition_transform(condition)
    kept_condition, kept_ranges = [], []
    dropped = 0
    started = time.monotonic()
    total = condition.shape[0]
    print(f"[label] start rows={total} sweep_steps={teacher.config.sweep_steps} "
          f"fold_samples={teacher.config.fold_samples} device={teacher.device}", flush=True)
    for index, (row, allowed) in enumerate(zip(condition, support), start=1):
        try:
            kept_ranges.append(teacher.label(row, allowed).cpu())
            kept_condition.append(row)
        except RuntimeError:
            dropped += 1
        if index % progress_every == 0 or index == total:
            elapsed = time.monotonic() - started
            rate = index / max(elapsed, 1e-9)
            eta = (total - index) / max(rate, 1e-9)
            print(f"[label] {index}/{total} valid={len(kept_condition)} dropped={dropped} "
                  f"rate={rate:.2f} rows/s eta={eta / 60.0:.1f} min", flush=True)
    if not kept_condition:
        raise RuntimeError("No valid labels in --support-dataset")
    return torch.stack(kept_condition), torch.stack(kept_ranges), dropped


def train(
    network: HaaRangeNetwork, x: Tensor, y: Tensor, *, steps: int,
    batch_size: int, lr: float, progress_every: int,
) -> float:
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    network.train()
    final = float("nan")
    for step in range(1, steps + 1):
        ids = torch.randint(x.shape[0], (min(batch_size, x.shape[0]),), device=x.device)
        prediction = network(x[ids])
        # Endpoint error is the deployed contract.  A small half-range term
        # makes narrow safe bands harder for the network to wash out.
        loss = torch.mean((prediction - y[ids]).square())
        pred_half = 0.5 * (prediction[..., 1] - prediction[..., 0])
        true_half = 0.5 * (y[ids, ..., 1] - y[ids, ..., 0])
        loss = loss + 0.25 * torch.mean((pred_half - true_half).square())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final = float(loss.detach().cpu())
        if step % progress_every == 0 or step == steps:
            print(f"[train] {step}/{steps} loss={final:.8f}", flush=True)
    network.eval()
    return final


@torch.no_grad()
def evaluate(
    network: HaaRangeNetwork, x: Tensor, y: Tensor, *, label_margin: float,
) -> dict[str, float | int]:
    """Measure endpoint fit and unsafe expansion beyond the teacher interval."""
    prediction = network(x)
    endpoint_error = (prediction - y).abs()
    # A lower bound below the teacher lower bound, or an upper bound above the
    # teacher upper bound, enlarges the permitted action set and is unsafe.
    lower_expansion = (y[..., 0] - prediction[..., 0]).clamp_min(0.0)
    upper_expansion = (prediction[..., 1] - y[..., 1]).clamp_min(0.0)
    unsafe = torch.maximum(lower_expansion, upper_expansion)
    # The teacher first contracted the geometric interval by label_margin.
    # An expansion smaller than this consumes the buffer but is not yet an
    # estimated geometric boundary crossing.
    remaining = label_margin - unsafe
    return {
        "endpoint_mae_rad": float(endpoint_error.mean().cpu()),
        "endpoint_max_rad": float(endpoint_error.max().cpu()),
        "unsafe_mean_expansion_rad": float(unsafe.mean().cpu()),
        "unsafe_max_expansion_rad": float(unsafe.max().cpu()),
        "unsafe_endpoint_count": int((unsafe > 1e-6).sum().cpu()),
        "endpoint_count": int(unsafe.numel()),
        "estimated_remaining_margin_rad": float(remaining.min().cpu()),
        "margin_exhausted_endpoint_count": int((remaining < 0.0).sum().cpu()),
    }


def split_train_validation(
    x: Tensor, y: Tensor, *, validation_fraction: float, seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if x.shape[0] < 10:
        raise ValueError("At least 10 valid labels are required for a train/validation split")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(x.shape[0], generator=generator)
    validation_count = max(1, round(x.shape[0] * validation_fraction))
    validation_count = min(validation_count, x.shape[0] - 1)
    validation_ids, train_ids = order[:validation_count], order[validation_count:]
    return x[train_ids], y[train_ids], x[validation_ids], y[validation_ids]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument(
        "--support-dataset", type=Path,
        help="optional .pt dict: condition [N,8], allowed_support [N,32]; preferred for real LiDAR envelopes",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synthetic-body-margin", type=float, default=0.45)
    parser.add_argument("--output-margin", type=float, default=0.06)
    # High-fidelity geometry defaults. They are intentionally expensive, but
    # are needed when this checkpoint will gate real HAA motion.
    parser.add_argument("--fold-samples", type=int, default=8192)
    parser.add_argument("--fold-bins", type=int, default=257)
    parser.add_argument("--sweep-steps", type=int, default=201)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--train-progress-every", type=int, default=1000)
    parser.add_argument("--max-label-rows", type=int, help="label only the first N dataset rows")
    parser.add_argument("--teacher-device", default="cpu", help="cpu or cuda; geometry-label device")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if (args.samples < 1 or args.steps < 1 or args.progress_every < 1
            or args.train_progress_every < 1):
        raise ValueError("--samples, --steps and progress intervals must be positive")
    torch.manual_seed(args.seed)

    spec = load_envelope_condition_spec()
    range_config = HaaRangeConfig(condition_names=spec.condition_names)
    teacher = PackageHaaRangeTeacher(
        range_config,
        config=PackageHaaTeacherConfig(
            synthetic_body_margin=args.synthetic_body_margin,
            output_margin=args.output_margin,
            fold_samples=args.fold_samples,
            fold_bins=args.fold_bins,
            sweep_steps=args.sweep_steps,
        ),
        device=args.teacher_device,
    )
    low, high = spec.bounds(torch.device("cpu"))
    transform = lambda rows: apply_env_morphology_priors(rows, spec)
    if args.support_dataset:
        x_cpu, y_cpu, dropped = make_dataset_from_support(
            teacher, args.support_dataset, condition_transform=transform,
            progress_every=args.progress_every, max_rows=args.max_label_rows,
        )
    else:
        x_cpu, y_cpu, dropped = make_dataset(
            teacher, low, high, samples=args.samples, condition_transform=transform,
            progress_every=args.progress_every,
        )
    x_train_cpu, y_train_cpu, x_validation_cpu, y_validation_cpu = split_train_validation(
        x_cpu, y_cpu, validation_fraction=args.validation_fraction, seed=args.seed,
    )
    print(f"[split] train={len(x_train_cpu)} validation={len(x_validation_cpu)} "
          f"seed={args.seed}", flush=True)
    device = torch.device(args.device)
    network = HaaRangeNetwork(range_config).to(device)
    loss = train(
        network, x_train_cpu.to(device), y_train_cpu.to(device), steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, progress_every=args.train_progress_every,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    network.save_checkpoint(args.output)
    train_metrics = evaluate(
        network, x_train_cpu.to(device), y_train_cpu.to(device), label_margin=args.output_margin,
    )
    validation_metrics = evaluate(
        network, x_validation_cpu.to(device), y_validation_cpu.to(device), label_margin=args.output_margin,
    )
    print(
        f"saved={args.output} valid={len(x_cpu)} dropped={dropped} train_loss={loss:.8f}\n"
        f"[train-metrics] mae={train_metrics['endpoint_mae_rad']:.6f} "
        f"max={train_metrics['endpoint_max_rad']:.6f} "
        f"unsafe_max={train_metrics['unsafe_max_expansion_rad']:.6f}\n"
        f"[validation] mae={validation_metrics['endpoint_mae_rad']:.6f} "
        f"max={validation_metrics['endpoint_max_rad']:.6f} "
        f"unsafe_mean={validation_metrics['unsafe_mean_expansion_rad']:.6f} "
        f"unsafe_max={validation_metrics['unsafe_max_expansion_rad']:.6f} "
        f"unsafe_endpoints={validation_metrics['unsafe_endpoint_count']}/"
        f"{validation_metrics['endpoint_count']} remaining_margin="
        f"{validation_metrics['estimated_remaining_margin_rad']:.6f} "
        f"margin_exhausted={validation_metrics['margin_exhausted_endpoint_count']}"
    )
    metadata = {
        "teacher": "el4090_envelope.haa_rejection_ranges",
        "input_contract": "condition [N,8], derived morphology priors",
        "output_contract": "legacy HAA order RF/RM/RB/LF/LM/LB, [N,6,2] lower/upper radians",
        "support_source": str(args.support_dataset) if args.support_dataset else "synthetic symmetric condition footprint",
        "valid_samples": len(x_cpu), "dropped_samples": dropped,
        "train_samples": len(x_train_cpu), "validation_samples": len(x_validation_cpu),
        "seed": args.seed, "validation_fraction": args.validation_fraction,
        "train_metrics": train_metrics, "validation_metrics": validation_metrics,
        "teacher_config": vars(teacher.config),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
