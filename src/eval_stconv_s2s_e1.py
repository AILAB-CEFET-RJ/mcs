#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained STConvS2S E1 checkpoint on val/test.

Outputs:
  - metrics_<split>.json
  - metrics_<split>.csv
  - predictions_<split>.npz

The prediction archive stores both continuous counts and final integer counts:
  pred_counts     (N, 28, 11, 21)
  pred_rounded    (N, 28, 11, 21)
  target          (N, 28, 11, 21)
  anchor_dates    (N,)
  horizons        (28,)
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
from models.stconv_s2s import (  # noqa: E402
    STConvS2SGrid,
    STConvS2SOfficialRGrid,
    STConvS2SOfficialRHurdleGrid,
    STConvS2SUpstreamRGrid,
    STConvS2SUpstreamRHurdleGrid,
    count_parameters,
)
from train_stconv_s2s_e1 import (  # noqa: E402
    assert_prediction_shape,
    compute_loss,
    compute_auto_pos_weight,
    finalize_metric_pair,
    make_active_mask,
    metric_accumulator,
    pred_to_counts,
    to_model_tensors,
    update_metric_pair,
)

LOG = logging.getLogger("eval_stconv_s2s_e1")


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(outdir / "eval.log", encoding="utf-8"),
        ],
    )


def build_model(config: dict) -> torch.nn.Module:
    model_name = config.get("model", "official-r")
    grid_h = int(config.get("grid_h", 11))
    grid_w = int(config.get("grid_w", 21))
    if model_name == "minimal":
        return STConvS2SGrid(
            in_channels=45,
            hidden_channels=int(config.get("hidden_channels", 16)),
            out_channels=1,
            t_in=28,
            t_out=28,
            temporal_layers=int(config.get("temporal_layers", 2)),
            spatial_layers=int(config.get("spatial_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if model_name == "official-r":
        return STConvS2SOfficialRGrid(
            in_channels=45,
            hidden_channels=int(config.get("hidden_channels", 16)),
            out_channels=1,
            t_in=28,
            t_out=28,
            num_layers=int(config.get("temporal_layers", 2)),
            kernel_size=3,
            dropout=float(config.get("dropout", 0.1)),
            height=grid_h,
            width=grid_w,
        )
    if model_name == "official-r-hurdle":
        return STConvS2SOfficialRHurdleGrid(
            in_channels=45,
            hidden_channels=int(config.get("hidden_channels", 16)),
            t_in=28,
            t_out=28,
            num_layers=int(config.get("temporal_layers", 2)),
            kernel_size=3,
            dropout=float(config.get("dropout", 0.1)),
            height=grid_h,
            width=grid_w,
        )
    if model_name == "upstream-r":
        return STConvS2SUpstreamRGrid(
            in_channels=45,
            hidden_channels=int(config.get("hidden_channels", 16)),
            out_channels=1,
            t_in=28,
            t_out=28,
            num_layers=int(config.get("temporal_layers", 2)),
            kernel_size=3,
            dropout=float(config.get("dropout", 0.1)),
            device=str(config.get("device", "cpu")),
            height=grid_h,
            width=grid_w,
        )
    if model_name == "upstream-r-hurdle":
        return STConvS2SUpstreamRHurdleGrid(
            in_channels=45,
            hidden_channels=int(config.get("hidden_channels", 16)),
            t_in=28,
            t_out=28,
            num_layers=int(config.get("temporal_layers", 2)),
            kernel_size=3,
            dropout=float(config.get("dropout", 0.1)),
            device=str(config.get("device", "cpu")),
            height=grid_h,
            width=grid_w,
        )
    raise ValueError(f"Modelo desconhecido no checkpoint: {model_name}")


def write_metrics(metrics: dict[str, float], outdir: Path, split: str) -> None:
    with open(outdir / f"metrics_{split}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / f"metrics_{split}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics))
        writer.writeheader()
        writer.writerow(metrics)


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, float]:
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    config = checkpoint.get("config", {})
    loss_name = args.loss or config.get("loss", "mse_log1p")

    outdir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / f"eval_{args.split}"
    setup_logging(outdir)

    LOG.info("Checkpoint: %s", checkpoint_path)
    LOG.info("Output dir: %s", outdir)
    LOG.info("Split: %s", args.split)
    LOG.info("Loss usada para avaliacao: %s", loss_name)

    device = torch.device(args.device)
    dataset_dir = args.dataset_dir or config.get("dataset_dir", "")
    train_ds = RJDengueCleanDataset(split="train", target_mode="daily28", dataset_dir=dataset_dir)
    eval_ds = RJDengueCleanDataset(split=args.split, target_mode="daily28", dataset_dir=dataset_dir)
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    active_mask = make_active_mask(train_ds, device)
    pos_weight_tensor = (
        compute_auto_pos_weight(train_ds, float(config.get("pos_weight", 0.0)), device)
        if loss_name == "hurdle_poisson"
        else None
    )
    occurrence_weight = float(config.get("occurrence_weight", 1.0))
    count_weight = float(config.get("count_weight", 1.0))
    occurrence_threshold = (
        float(args.occurrence_threshold)
        if args.occurrence_threshold >= 0.0
        else float(config.get("occurrence_threshold", 0.5))
    )

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    LOG.info("Modelo: %s", model.__class__.__name__)
    LOG.info("Parametros treinaveis: %d", count_parameters(model))
    LOG.info("Occurrence threshold: %.6f", occurrence_threshold)

    acc_continuous = metric_accumulator()
    acc_rounded = metric_accumulator()
    losses: list[float] = []
    pred_chunks: list[np.ndarray] = []
    rounded_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    anchor_dates: list[str] = []

    for i, batch in enumerate(loader, start=1):
        x, target = to_model_tensors(batch, device)
        pred = model(x)
        assert_prediction_shape(pred, target)

        loss = compute_loss(pred, target, loss_name, occurrence_weight, count_weight, pos_weight_tensor)
        losses.append(float(loss.detach().cpu()))

        pred_counts = pred_to_counts(pred, loss_name, occurrence_threshold)
        pred_rounded = torch.round(pred_counts).clamp_min(0.0)
        update_metric_pair(acc_continuous, acc_rounded, pred_counts, target, active_mask)

        pred_chunks.append(pred_counts.squeeze(1).detach().cpu().numpy().astype(np.float32))
        rounded_chunks.append(pred_rounded.squeeze(1).detach().cpu().numpy().astype(np.int16))
        target_chunks.append(target.squeeze(1).detach().cpu().numpy().astype(np.float32))
        anchor_dates.extend([str(d) for d in batch["anchor_date"]])

        if args.max_batches > 0 and i >= args.max_batches:
            break

    metrics = finalize_metric_pair(acc_continuous, acc_rounded, args.split)
    metrics[f"{args.split}_loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["n_samples"] = int(sum(x.shape[0] for x in pred_chunks))
    metrics["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    write_metrics(metrics, outdir, args.split)

    pred_arr = np.concatenate(pred_chunks, axis=0)
    rounded_arr = np.concatenate(rounded_chunks, axis=0)
    target_arr = np.concatenate(target_chunks, axis=0)
    np.savez_compressed(
        outdir / f"predictions_{args.split}.npz",
        pred_counts=pred_arr,
        pred_rounded=rounded_arr,
        target=target_arr,
        anchor_dates=np.asarray(anchor_dates),
        horizons=np.arange(1, 29, dtype=np.int16),
    )

    LOG.info("Amostras avaliadas: %d", metrics["n_samples"])
    LOG.info("%s_loss=%.6f", args.split, metrics[f"{args.split}_loss"])
    LOG.info("%s_mae_active=%.6f", args.split, metrics[f"{args.split}_mae_all28_activecells"])
    LOG.info("%s_rounded_mae_active=%.6f", args.split, metrics[f"{args.split}_rounded_mae_all28_activecells"])
    LOG.info("Predicoes salvas: %s", outdir / f"predictions_{args.split}.npz")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STConvS2S E1 checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path para best.pt ou last.pt")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-dir", default="", help="Opcional: sobrescreve dataset_dir do checkpoint")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss", default="", help="Opcional: sobrescreve a loss do checkpoint")
    parser.add_argument(
        "--occurrence-threshold",
        type=float,
        default=-1.0,
        help="Opcional para hurdle: sobrescreve threshold do checkpoint; -1 usa config",
    )
    parser.add_argument("--max-batches", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    evaluate_checkpoint(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
