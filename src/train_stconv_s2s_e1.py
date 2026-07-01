#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/smoke-test STConvS2SGrid on RJ_E1_T1_clean.

The script intentionally starts as a detailed, conservative training harness:
it validates tensor contracts, supports a tiny overfit mode, writes metrics and
checkpoints, and keeps the dataset contract explicit.

Example smoke test:
    pipenv run python arboseer\\src\\train_stconv_s2s_e1.py --epochs 1 --batch-size 1 --overfit-batches 1 --hidden-channels 8

Example fuller run:
    pipenv run python arboseer\\src\\train_stconv_s2s_e1.py --epochs 50 --batch-size 8 --hidden-channels 32
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "arboseer" / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.rj_dengue_clean_dataset import RJDengueCleanDataset  # noqa: E402
from models.stconv_s2s import STConvS2SGrid, count_parameters  # noqa: E402

LOG = logging.getLogger("train_stconv_s2s_e1")

EXPECTED_X_DATASET = (28, 45, 11, 21)
EXPECTED_Y_DATASET = (28, 11, 21)
EVAL_HORIZONS = [1, 7, 14, 21, 28]


@dataclass
class RunConfig:
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_channels: int
    temporal_layers: int
    spatial_layers: int
    dropout: float
    loss: str
    device: str
    num_workers: int
    overfit_batches: int
    max_train_batches: int
    max_val_batches: int
    output_dir: str


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_contract_assertions(batch: dict) -> None:
    x = batch["X"]
    y = batch["Y"]
    if tuple(x.shape[1:]) != EXPECTED_X_DATASET:
        raise AssertionError(f"Contrato X dataset quebrado: esperado (*,{EXPECTED_X_DATASET}), veio {tuple(x.shape)}")
    if tuple(y.shape[1:]) != EXPECTED_Y_DATASET:
        raise AssertionError(f"Contrato Y dataset quebrado: esperado (*,{EXPECTED_Y_DATASET}), veio {tuple(y.shape)}")
    if torch.isnan(x).any():
        raise AssertionError("X contem NaN")
    if torch.isnan(y).any():
        raise AssertionError("Y contem NaN")


def to_model_tensors(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    collate_contract_assertions(batch)
    x = batch["X"].to(device=device, dtype=torch.float32)  # (B, T, C, H, W)
    y = batch["Y"].to(device=device, dtype=torch.float32)  # (B, T, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()             # (B, C, T, H, W)
    y = y.unsqueeze(1).contiguous()                       # (B, 1, T, H, W)
    return x, y


def make_active_mask(train_dataset: RJDengueCleanDataset, device: torch.device) -> torch.Tensor:
    active = (train_dataset._targets.sum(axis=0) > 0).astype(np.float32)  # noqa: SLF001
    mask = torch.from_numpy(active)[None, None, None].to(device=device)
    LOG.info("Mascara ativa: %d/%d celulas", int(active.sum()), active.size)
    return mask


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse_log1p":
        target_log = torch.log1p(target)
        return F.mse_loss(pred, target_log)
    if loss_name == "mae_log1p":
        target_log = torch.log1p(target)
        return F.l1_loss(pred, target_log)
    if loss_name == "mse_raw":
        return F.mse_loss(pred, target)
    if loss_name == "mae_raw":
        return F.l1_loss(pred, target)
    raise ValueError(f"loss desconhecida: {loss_name}")


def pred_to_counts(pred: torch.Tensor, loss_name: str) -> torch.Tensor:
    if "log1p" in loss_name:
        return torch.expm1(pred).clamp_min(0.0)
    return pred.clamp_min(0.0)


def metric_accumulator() -> dict:
    return {
        "sum_abs_all": 0.0,
        "sum_sq_all": 0.0,
        "n_all": 0.0,
        "sum_abs_active": 0.0,
        "sum_sq_active": 0.0,
        "n_active": 0.0,
        "horizon_abs": {h: 0.0 for h in EVAL_HORIZONS},
        "horizon_n": {h: 0.0 for h in EVAL_HORIZONS},
    }


def update_metrics(acc: dict, pred_counts: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> None:
    diff = pred_counts - target
    abs_diff = diff.abs()
    sq_diff = diff.square()

    acc["sum_abs_all"] += float(abs_diff.sum().detach().cpu())
    acc["sum_sq_all"] += float(sq_diff.sum().detach().cpu())
    acc["n_all"] += float(abs_diff.numel())

    active = active_mask.expand_as(abs_diff)
    acc["sum_abs_active"] += float((abs_diff * active).sum().detach().cpu())
    acc["sum_sq_active"] += float((sq_diff * active).sum().detach().cpu())
    acc["n_active"] += float(active.sum().detach().cpu())

    for h in EVAL_HORIZONS:
        idx = h - 1
        h_abs = abs_diff[:, :, idx]
        acc["horizon_abs"][h] += float(h_abs.sum().detach().cpu())
        acc["horizon_n"][h] += float(h_abs.numel())


def finalize_metrics(acc: dict, prefix: str) -> dict[str, float]:
    out = {
        f"{prefix}_mae_all28_allcells": acc["sum_abs_all"] / max(acc["n_all"], 1.0),
        f"{prefix}_rmse_all28_allcells": math.sqrt(acc["sum_sq_all"] / max(acc["n_all"], 1.0)),
        f"{prefix}_mae_all28_activecells": acc["sum_abs_active"] / max(acc["n_active"], 1.0),
        f"{prefix}_rmse_all28_activecells": math.sqrt(acc["sum_sq_active"] / max(acc["n_active"], 1.0)),
    }
    for h in EVAL_HORIZONS:
        out[f"{prefix}_mae_h{h}_allcells"] = acc["horizon_abs"][h] / max(acc["horizon_n"][h], 1.0)
    return out


def maybe_limited(loader: Iterable, max_batches: int) -> Iterable:
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        yield batch


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
    active_mask: torch.Tensor,
    max_batches: int,
) -> dict[str, float]:
    model.train()
    acc = metric_accumulator()
    losses: list[float] = []
    for batch in maybe_limited(loader, max_batches):
        x, target = to_model_tensors(batch, device)
        pred = model(x)
        if tuple(pred.shape) != tuple(target.shape):
            raise AssertionError(f"Modelo retornou {tuple(pred.shape)}, esperado {tuple(target.shape)}")

        loss = compute_loss(pred, target, loss_name)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        update_metrics(acc, pred_to_counts(pred.detach(), loss_name), target, active_mask)

    metrics = finalize_metrics(acc, "train")
    metrics["train_loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    active_mask: torch.Tensor,
    max_batches: int,
    prefix: str,
) -> dict[str, float]:
    model.eval()
    acc = metric_accumulator()
    losses: list[float] = []
    for batch in maybe_limited(loader, max_batches):
        x, target = to_model_tensors(batch, device)
        pred = model(x)
        loss = compute_loss(pred, target, loss_name)
        losses.append(float(loss.detach().cpu()))
        update_metrics(acc, pred_to_counts(pred, loss_name), target, active_mask)

    metrics = finalize_metrics(acc, prefix)
    metrics[f"{prefix}_loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    keys = ["epoch"] + sorted(k for k in rows[0] if k != "epoch")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, RJDengueCleanDataset]:
    train_ds = RJDengueCleanDataset(split="train", target_mode="daily28")
    val_ds = RJDengueCleanDataset(split="val", target_mode="daily28")

    if args.overfit_batches > 0:
        n = min(len(train_ds), args.overfit_batches * args.batch_size)
        indices = list(range(n))
        LOG.info("Modo overfit: usando as primeiras %d amostras de treino também como validação", n)
        train_data = Subset(train_ds, indices)
        val_data = Subset(train_ds, indices)
    else:
        train_data = train_ds
        val_data = val_ds

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STConvS2SGrid on RJ_E1_T1_clean daily28")
    parser.add_argument("--seed", type=int, default=987)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--loss", choices=["mse_log1p", "mae_log1p", "mse_raw", "mae_raw"], default="mse_log1p")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overfit-batches", type=int, default=0, help=">0 usa poucos batches para sanity/overfit")
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    if args.output_dir:
        outdir = Path(args.output_dir)
    else:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "overfit" if args.overfit_batches > 0 else "train"
        outdir = PROJECT_ROOT / "arboseer" / "models" / f"RJ_E1_T1_clean_stconv_s2s_{mode}_{tag}"
    setup_logging(outdir)

    cfg = RunConfig(**{k: getattr(args, k) for k in RunConfig.__annotations__})
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    LOG.info("Output dir: %s", outdir)
    LOG.info("Device solicitado: %s", args.device)
    device = torch.device(args.device)

    train_loader, val_loader, train_ds = build_loaders(args)
    active_mask = make_active_mask(train_ds, device)

    model = STConvS2SGrid(
        in_channels=45,
        hidden_channels=args.hidden_channels,
        out_channels=1,
        t_in=28,
        t_out=28,
        temporal_layers=args.temporal_layers,
        spatial_layers=args.spatial_layers,
        dropout=args.dropout,
    ).to(device)
    LOG.info("Modelo: %s", model.__class__.__name__)
    LOG.info("Parametros treinaveis: %d", count_parameters(model))

    first_batch = next(iter(train_loader))
    x0, y0 = to_model_tensors(first_batch, device)
    with torch.no_grad():
        p0 = model(x0)
    LOG.info("Contrato modelo: X %s -> pred %s; target %s", tuple(x0.shape), tuple(p0.shape), tuple(y0.shape))
    if tuple(p0.shape) != tuple(y0.shape):
        raise AssertionError("Contrato de saida do modelo nao bate com target")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_rows: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.loss,
            active_mask,
            args.max_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.loss,
            active_mask,
            args.max_val_batches,
            prefix="val",
        )
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        metrics_rows.append(row)
        write_metrics_csv(outdir / "metrics.csv", metrics_rows)

        LOG.info(
            "epoch=%03d train_loss=%.6f val_loss=%.6f val_mae_active=%.4f val_mae_h28=%.4f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_mae_all28_activecells"],
            row["val_mae_h28_allcells"],
        )

        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, outdir / "last.pt")
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, outdir / "best.pt")

    LOG.info("Concluido. best_val_loss=%.6f", best_val)
    LOG.info("Artefatos: %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

