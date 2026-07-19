#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate simple E1 grid baselines on RJ_E1_T1_clean.

Currently implemented:
  - persistence: repeats cases at anchor day for all 28 future days.
  - moving_average_7: repeats the 7-day mean ending at anchor day.
  - moving_average_28: repeats the 28-day mean ending at anchor day.

Outputs:
  - metrics_<split>_<baseline>.json
  - metrics_<split>_<baseline>.csv
  - predictions_<split>_<baseline>.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "arboseer" / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.rj_dengue_clean_dataset import RJDengueCleanDataset  # noqa: E402
from train_stconv_s2s_e1 import (  # noqa: E402
    finalize_metric_pair,
    make_active_mask,
    metric_accumulator,
    update_metric_pair,
)

LOG = logging.getLogger("eval_e1_baselines")


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(outdir / "eval_baselines.log", encoding="utf-8"),
        ],
    )


def write_metrics(metrics: dict[str, float], outdir: Path, split: str, baseline: str) -> None:
    with open(outdir / f"metrics_{split}_{baseline}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / f"metrics_{split}_{baseline}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def build_baseline_prediction(
    baseline: str,
    targets_daily: np.ndarray,
    batch_anchors: np.ndarray,
) -> np.ndarray:
    """Return baseline prediction with shape (B, 28, 11, 21)."""
    if baseline == "persistence":
        anchor_maps = np.stack([targets_daily[int(t)] for t in batch_anchors], axis=0)
        return np.repeat(anchor_maps[:, None, :, :], 28, axis=1).astype(np.float32)
    if baseline == "moving_average_7":
        maps = []
        for t in batch_anchors:
            t = int(t)
            hist = targets_daily[t - 6 : t + 1]
            maps.append(hist.mean(axis=0))
        anchor_maps = np.stack(maps, axis=0)
        return np.repeat(anchor_maps[:, None, :, :], 28, axis=1).astype(np.float32)
    if baseline == "moving_average_28":
        maps = []
        for t in batch_anchors:
            t = int(t)
            hist = targets_daily[t - 27 : t + 1]
            maps.append(hist.mean(axis=0))
        anchor_maps = np.stack(maps, axis=0)
        return np.repeat(anchor_maps[:, None, :, :], 28, axis=1).astype(np.float32)
    raise ValueError(f"baseline desconhecido: {baseline}")


def evaluate_baseline(args: argparse.Namespace) -> dict[str, float]:
    baseline = args.baseline
    dataset_name = Path(args.dataset_dir).name if args.dataset_dir else "RJ_E1_T1_clean"
    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "arboseer" / "models" / f"{dataset_name}_baselines"
    )
    setup_logging(outdir)
    device = torch.device("cpu")

    train_ds = RJDengueCleanDataset(split="train", target_mode="daily28", dataset_dir=args.dataset_dir)
    ds = RJDengueCleanDataset(split=args.split, target_mode="daily28", dataset_dir=args.dataset_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    active_mask = make_active_mask(train_ds, device)

    acc_continuous = metric_accumulator()
    acc_rounded = metric_accumulator()
    pred_chunks: list[np.ndarray] = []
    rounded_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    anchor_dates: list[str] = []

    # Access to anchors/targets is deliberate for baselines: persistence must use
    # the true observed count at anchor day, not the scaled CASES input channel.
    anchors = ds._anchors  # noqa: SLF001
    targets_daily = ds._targets  # noqa: SLF001
    dates = ds._dates  # noqa: SLF001
    cursor = 0

    for i, batch in enumerate(loader, start=1):
        b = batch["Y"].shape[0]
        batch_anchors = anchors[cursor : cursor + b]
        cursor += b

        y_true = batch["Y"].to(dtype=torch.float32)  # (B, 28, 11, 21)
        pred = build_baseline_prediction(baseline, targets_daily, batch_anchors)
        pred_t = torch.from_numpy(pred)
        target_t = y_true

        update_metric_pair(
            acc_continuous,
            acc_rounded,
            pred_t.unsqueeze(1),
            target_t.unsqueeze(1),
            active_mask,
        )

        pred_chunks.append(pred)
        rounded_chunks.append(np.round(pred).clip(min=0).astype(np.int16))
        target_chunks.append(target_t.numpy().astype(np.float32))
        anchor_dates.extend([str(dates[int(t)]) for t in batch_anchors])

        if args.max_batches > 0 and i >= args.max_batches:
            break

    metrics = finalize_metric_pair(acc_continuous, acc_rounded, args.split)
    metrics["n_samples"] = int(sum(x.shape[0] for x in pred_chunks))
    metrics["baseline"] = baseline
    write_metrics(metrics, outdir, args.split, baseline)

    pred_arr = np.concatenate(pred_chunks, axis=0)
    rounded_arr = np.concatenate(rounded_chunks, axis=0)
    target_arr = np.concatenate(target_chunks, axis=0)
    np.savez_compressed(
        outdir / f"predictions_{args.split}_{baseline}.npz",
        pred_counts=pred_arr,
        pred_rounded=rounded_arr,
        target=target_arr,
        anchor_dates=np.asarray(anchor_dates),
        horizons=np.arange(1, 29, dtype=np.int16),
    )

    LOG.info("Baseline: %s", baseline)
    LOG.info("Split: %s", args.split)
    LOG.info("Amostras avaliadas: %d", metrics["n_samples"])
    LOG.info("%s_mae_active=%.6f", args.split, metrics[f"{args.split}_mae_all28_activecells"])
    LOG.info("%s_rounded_mae_active=%.6f", args.split, metrics[f"{args.split}_rounded_mae_all28_activecells"])
    LOG.info("Predicoes salvas: %s", outdir / f"predictions_{args.split}_{baseline}.npz")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E1 baselines")
    parser.add_argument(
        "--baseline",
        choices=["persistence", "moving_average_7", "moving_average_28"],
        default="persistence",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--dataset-dir",
        default=str(PROJECT_ROOT / "arboseer" / "data" / "datasets" / "RJ_E1_T1_clean"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
