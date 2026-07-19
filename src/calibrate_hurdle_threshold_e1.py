#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate the occurrence threshold for STConvS2S hurdle checkpoints.

The hurdle model predicts two tensors:
  - occ_logits: logits for P(y > 0)
  - log_lambda: Poisson log-rate for counts

Training currently uses a fixed threshold of 0.5 when converting probabilities
to final counts. This script chooses that threshold on a validation split
without retraining the model.
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
from eval_stconv_s2s_e1 import build_model  # noqa: E402
from train_stconv_s2s_e1 import assert_prediction_shape, to_model_tensors  # noqa: E402

LOG = logging.getLogger("calibrate_hurdle_threshold_e1")
EVAL_HORIZONS = [1, 7, 14, 21, 28]


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(outdir / "threshold_calibration.log", encoding="utf-8"),
        ],
    )


def active_mask_from_train(dataset_dir: str) -> np.ndarray:
    train_ds = RJDengueCleanDataset(split="train", target_mode="daily28", dataset_dir=dataset_dir)
    active = (train_ds._targets.sum(axis=0) > 0).astype(bool)  # noqa: SLF001
    LOG.info("Mascara ativa: %d/%d celulas", int(active.sum()), active.size)
    return active


@torch.no_grad()
def collect_hurdle_outputs(
    checkpoint_path: Path,
    split: str,
    batch_size: int,
    device_name: str,
    num_workers: int,
    max_batches: int,
    dataset_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device_name, weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("loss") != "hurdle_poisson":
        raise ValueError(f"Checkpoint nao parece hurdle_poisson: loss={config.get('loss')}")

    device = torch.device(device_name)
    dataset = RJDengueCleanDataset(split=split, target_mode="daily28", dataset_dir=dataset_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    occ_chunks: list[np.ndarray] = []
    lambda_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    anchor_dates: list[str] = []

    for i, batch in enumerate(loader, start=1):
        x, target = to_model_tensors(batch, device)
        pred = model(x)
        assert_prediction_shape(pred, target)
        if not isinstance(pred, dict):
            raise TypeError("Checkpoint hurdle deveria retornar dict com occ_logits e log_lambda")

        occ_prob = torch.sigmoid(pred["occ_logits"])
        lambda_counts = torch.exp(pred["log_lambda"]).clamp_min(0.0)

        occ_chunks.append(occ_prob.squeeze(1).detach().cpu().numpy().astype(np.float32))
        lambda_chunks.append(lambda_counts.squeeze(1).detach().cpu().numpy().astype(np.float32))
        target_chunks.append(target.squeeze(1).detach().cpu().numpy().astype(np.float32))
        anchor_dates.extend([str(d) for d in batch["anchor_date"]])

        if max_batches > 0 and i >= max_batches:
            break

    return (
        np.concatenate(occ_chunks, axis=0),
        np.concatenate(lambda_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
        anchor_dates,
        config,
        int(checkpoint.get("epoch", -1)),
    )


def compute_metrics(pred_counts: np.ndarray, target: np.ndarray, active_mask: np.ndarray, prefix: str) -> dict[str, float]:
    diff = pred_counts - target
    abs_diff = np.abs(diff)
    sq_diff = diff * diff

    active = active_mask[None, None, :, :]
    n_active = float(active.sum() * pred_counts.shape[0] * pred_counts.shape[1])

    metrics = {
        f"{prefix}_mae_all28_allcells": float(abs_diff.mean()),
        f"{prefix}_rmse_all28_allcells": float(np.sqrt(sq_diff.mean())),
        f"{prefix}_mae_all28_activecells": float((abs_diff * active).sum() / max(n_active, 1.0)),
        f"{prefix}_rmse_all28_activecells": float(np.sqrt((sq_diff * active).sum() / max(n_active, 1.0))),
    }
    for h in EVAL_HORIZONS:
        metrics[f"{prefix}_mae_h{h}_allcells"] = float(abs_diff[:, h - 1].mean())
    return metrics


def evaluate_threshold(
    occ_prob: np.ndarray,
    lambda_counts: np.ndarray,
    target: np.ndarray,
    active_mask: np.ndarray,
    threshold: float,
    split: str,
) -> dict[str, float]:
    pred_counts = np.where(occ_prob >= threshold, lambda_counts, 0.0).astype(np.float32)
    rounded = np.rint(pred_counts).clip(min=0.0)
    metrics = {"threshold": float(threshold)}
    metrics.update(compute_metrics(pred_counts, target, active_mask, split))
    metrics.update(compute_metrics(rounded, target, active_mask, f"{split}_rounded"))
    return metrics


def write_rows(rows: list[dict[str, float]], outdir: Path, split: str) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with open(outdir / f"threshold_sweep_{split}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(outdir / f"threshold_sweep_{split}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def parse_thresholds(args: argparse.Namespace) -> list[float]:
    if args.thresholds:
        return [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    values = np.arange(args.min_threshold, args.max_threshold + args.step_threshold / 2.0, args.step_threshold)
    return [round(float(x), 6) for x in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate hurdle occurrence threshold for E1")
    parser.add_argument("--checkpoint", required=True, help="Path para best.pt/best_loss.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir", default="", help="Default: <checkpoint_dir>/threshold_calibration_<split>")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-dir", default="", help="Opcional: sobrescreve dataset_dir do checkpoint")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--min-threshold", type=float, default=0.01)
    parser.add_argument("--max-threshold", type=float, default=0.99)
    parser.add_argument("--step-threshold", type=float, default=0.01)
    parser.add_argument("--thresholds", default="", help="Lista explicita, ex: 0.05,0.10,0.20")
    parser.add_argument(
        "--selection-metric",
        default="rounded_mae_all28_activecells",
        choices=["rounded_mae_all28_activecells", "rounded_mae_all28_allcells", "rounded_rmse_all28_activecells"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    outdir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / f"threshold_calibration_{args.split}"
    setup_logging(outdir)

    LOG.info("Checkpoint: %s", checkpoint_path)
    LOG.info("Split: %s", args.split)
    LOG.info("Output dir: %s", outdir)
    checkpoint_for_config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint_for_config.get("config", {})
    dataset_dir = args.dataset_dir or config.get("dataset_dir", "")
    LOG.info("Dataset dir: %s", dataset_dir)

    active_mask = active_mask_from_train(dataset_dir)
    occ_prob, lambda_counts, target, anchor_dates, config, checkpoint_epoch = collect_hurdle_outputs(
        checkpoint_path=checkpoint_path,
        split=args.split,
        batch_size=args.batch_size,
        device_name=args.device,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        dataset_dir=dataset_dir,
    )

    LOG.info("Amostras: %d", occ_prob.shape[0])
    LOG.info("Checkpoint epoch: %d", checkpoint_epoch)

    rows = [
        evaluate_threshold(occ_prob, lambda_counts, target, active_mask, threshold, args.split)
        for threshold in parse_thresholds(args)
    ]

    metric_key = f"{args.split}_{args.selection_metric}"
    best = min(rows, key=lambda row: row[metric_key])
    best.update(
        {
            "split": args.split,
            "selection_metric": metric_key,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "n_samples": int(occ_prob.shape[0]),
            "model": str(config.get("model", "")),
        }
    )

    write_rows(rows, outdir, args.split)
    with open(outdir / f"best_threshold_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    LOG.info("Melhor threshold: %.6f", best["threshold"])
    LOG.info("%s=%.6f", metric_key, best[metric_key])
    LOG.info("Resumo salvo: %s", outdir / f"best_threshold_{args.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
