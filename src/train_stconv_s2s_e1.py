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
from models.stconv_s2s import (  # noqa: E402
    STConvS2SGrid,
    STConvS2SOfficialRGrid,
    STConvS2SOfficialRHurdleGrid,
    STConvS2SUpstreamRGrid,
    STConvS2SUpstreamRHurdleGrid,
    count_parameters,
)

LOG = logging.getLogger("train_stconv_s2s_e1")

EXPECTED_T_IN = 28
EXPECTED_CHANNELS = 45
EXPECTED_T_OUT = 28
EVAL_HORIZONS = [1, 7, 14, 21, 28]


@dataclass
class RunConfig:
    model: str
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
    checkpoint_metric: str
    occurrence_weight: float
    count_weight: float
    occurrence_threshold: float
    pos_weight: float
    dataset_dir: str
    grid_h: int
    grid_w: int
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
    if x.ndim != 5:
        raise AssertionError(f"Contrato X dataset quebrado: esperado 5D, veio {tuple(x.shape)}")
    if y.ndim != 4:
        raise AssertionError(f"Contrato Y dataset quebrado: esperado 4D, veio {tuple(y.shape)}")
    if x.shape[1] != EXPECTED_T_IN or x.shape[2] != EXPECTED_CHANNELS:
        raise AssertionError(
            f"Contrato X dataset quebrado: esperado T={EXPECTED_T_IN}, C={EXPECTED_CHANNELS}; veio {tuple(x.shape)}"
        )
    if y.shape[1] != EXPECTED_T_OUT:
        raise AssertionError(f"Contrato Y dataset quebrado: esperado T_out={EXPECTED_T_OUT}; veio {tuple(y.shape)}")
    if tuple(x.shape[-2:]) != tuple(y.shape[-2:]):
        raise AssertionError(f"Grid X/Y desalinhado: X {tuple(x.shape)}, Y {tuple(y.shape)}")
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


def compute_loss(
    pred,
    target: torch.Tensor,
    loss_name: str,
    occurrence_weight: float = 1.0,
    count_weight: float = 1.0,
    pos_weight_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_name == "hurdle_poisson":
        if not isinstance(pred, dict):
            raise TypeError("hurdle_poisson espera pred dict com occ_logits e log_lambda")
        occ_target = (target > 0).to(dtype=target.dtype)
        bce = F.binary_cross_entropy_with_logits(
            pred["occ_logits"],
            occ_target,
            pos_weight=pos_weight_tensor,
        )
        positive_mask = target > 0
        if positive_mask.any():
            count_loss = F.poisson_nll_loss(
                pred["log_lambda"][positive_mask],
                target[positive_mask],
                log_input=True,
                full=False,
            )
        else:
            count_loss = pred["log_lambda"].sum() * 0.0
        return occurrence_weight * bce + count_weight * count_loss
    if loss_name == "poisson_nll_log":
        return F.poisson_nll_loss(pred, target, log_input=True, full=False)
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


def pred_to_counts(pred, loss_name: str, occurrence_threshold: float = 0.5) -> torch.Tensor:
    if loss_name == "hurdle_poisson":
        if not isinstance(pred, dict):
            raise TypeError("hurdle_poisson espera pred dict com occ_logits e log_lambda")
        occ_prob = torch.sigmoid(pred["occ_logits"])
        counts = torch.exp(pred["log_lambda"]).clamp_min(0.0)
        return torch.where(occ_prob >= occurrence_threshold, counts, torch.zeros_like(counts))
    if loss_name == "poisson_nll_log":
        return torch.exp(pred).clamp_min(0.0)
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


def update_metric_pair(
    acc_continuous: dict,
    acc_rounded: dict,
    pred_counts: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> None:
    update_metrics(acc_continuous, pred_counts, target, active_mask)
    pred_rounded = torch.round(pred_counts).clamp_min(0.0)
    update_metrics(acc_rounded, pred_rounded, target, active_mask)


def finalize_metric_pair(acc_continuous: dict, acc_rounded: dict, prefix: str) -> dict[str, float]:
    metrics = finalize_metrics(acc_continuous, prefix)
    rounded = finalize_metrics(acc_rounded, f"{prefix}_rounded")
    metrics.update(rounded)
    return metrics


def maybe_limited(loader: Iterable, max_batches: int) -> Iterable:
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        yield batch


def assert_prediction_shape(pred, target: torch.Tensor) -> None:
    if isinstance(pred, dict):
        for key in ("occ_logits", "log_lambda"):
            if key not in pred:
                raise AssertionError(f"Predicao hurdle sem chave {key}")
            if tuple(pred[key].shape) != tuple(target.shape):
                raise AssertionError(f"Predicao {key} retornou {tuple(pred[key].shape)}, esperado {tuple(target.shape)}")
        return
    if tuple(pred.shape) != tuple(target.shape):
        raise AssertionError(f"Modelo retornou {tuple(pred.shape)}, esperado {tuple(target.shape)}")


def compute_auto_pos_weight(train_dataset: RJDengueCleanDataset, requested: float, device: torch.device) -> torch.Tensor | None:
    if requested < 0:
        return None
    if requested > 0:
        value = float(requested)
    else:
        y = train_dataset._targets  # noqa: SLF001
        positives = float((y > 0).sum())
        total = float(y.size)
        negatives = total - positives
        value = negatives / max(positives, 1.0)
    LOG.info("BCE pos_weight=%.4f", value)
    return torch.tensor(value, dtype=torch.float32, device=device)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
    active_mask: torch.Tensor,
    max_batches: int,
    occurrence_weight: float,
    count_weight: float,
    occurrence_threshold: float,
    pos_weight_tensor: torch.Tensor | None,
) -> dict[str, float]:
    model.train()
    acc_continuous = metric_accumulator()
    acc_rounded = metric_accumulator()
    losses: list[float] = []
    for batch in maybe_limited(loader, max_batches):
        x, target = to_model_tensors(batch, device)
        pred = model(x)
        assert_prediction_shape(pred, target)

        loss = compute_loss(pred, target, loss_name, occurrence_weight, count_weight, pos_weight_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        if isinstance(pred, dict):
            pred_detached = {k: v.detach() for k, v in pred.items()}
        else:
            pred_detached = pred.detach()
        pred_counts = pred_to_counts(pred_detached, loss_name, occurrence_threshold)
        update_metric_pair(acc_continuous, acc_rounded, pred_counts, target, active_mask)

    metrics = finalize_metric_pair(acc_continuous, acc_rounded, "train")
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
    occurrence_weight: float = 1.0,
    count_weight: float = 1.0,
    occurrence_threshold: float = 0.5,
    pos_weight_tensor: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    acc_continuous = metric_accumulator()
    acc_rounded = metric_accumulator()
    losses: list[float] = []
    for batch in maybe_limited(loader, max_batches):
        x, target = to_model_tensors(batch, device)
        pred = model(x)
        assert_prediction_shape(pred, target)
        loss = compute_loss(pred, target, loss_name, occurrence_weight, count_weight, pos_weight_tensor)
        losses.append(float(loss.detach().cpu()))
        pred_counts = pred_to_counts(pred, loss_name, occurrence_threshold)
        update_metric_pair(acc_continuous, acc_rounded, pred_counts, target, active_mask)

    metrics = finalize_metric_pair(acc_continuous, acc_rounded, prefix)
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
    train_ds = RJDengueCleanDataset(split="train", target_mode="daily28", dataset_dir=args.dataset_dir)
    val_ds = RJDengueCleanDataset(split="val", target_mode="daily28", dataset_dir=args.dataset_dir)

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
    parser.add_argument(
        "--model",
        choices=["minimal", "official-r", "official-r-hurdle", "upstream-r", "upstream-r-hurdle"],
        default="official-r",
        help=(
            "minimal = sanity baseline local; official-r = port local; "
            "upstream-r = usa external/stconvs2s; variantes hurdle = binary+count heads"
        ),
    )
    parser.add_argument("--seed", type=int, default=987)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--loss",
        choices=["mse_log1p", "mae_log1p", "mse_raw", "mae_raw", "poisson_nll_log", "hurdle_poisson"],
        default="mse_log1p",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dataset-dir",
        default=str(PROJECT_ROOT / "arboseer" / "data" / "datasets" / "RJ_E1_T1_clean"),
        help="Diretorio do dataset clean, ex: arboseer/data/datasets/RJ_E2_T1_clean",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overfit-batches", type=int, default=0, help=">0 usa poucos batches para sanity/overfit")
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 = sem limite")
    parser.add_argument(
        "--checkpoint-metric",
        default="val_rounded_mae_all28_activecells",
        help="Metrica de validacao minimizada para salvar best.pt",
    )
    parser.add_argument("--occurrence-weight", type=float, default=1.0, help="Peso da BCE no modelo hurdle")
    parser.add_argument("--count-weight", type=float, default=1.0, help="Peso da Poisson nos positivos no modelo hurdle")
    parser.add_argument("--occurrence-threshold", type=float, default=0.5, help="Threshold P(y>0) para predição final hurdle")
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=0.0,
        help="BCE pos_weight: 0=auto neg/pos; >0 valor fixo; <0 desativa",
    )
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
        dataset_name = Path(args.dataset_dir).name or "RJ_E1_T1_clean"
        outdir = PROJECT_ROOT / "arboseer" / "models" / f"{dataset_name}_stconv_s2s_{mode}_{tag}"
    setup_logging(outdir)

    LOG.info("Output dir: %s", outdir)
    LOG.info("Device solicitado: %s", args.device)
    LOG.info("Dataset dir: %s", args.dataset_dir)
    device = torch.device(args.device)

    train_loader, val_loader, train_ds = build_loaders(args)
    grid_h, grid_w = train_ds.grid_shape
    args.grid_h = int(grid_h)
    args.grid_w = int(grid_w)
    LOG.info("Grid dataset: %dx%d", args.grid_h, args.grid_w)

    cfg = RunConfig(**{k: getattr(args, k) for k in RunConfig.__annotations__})
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    active_mask = make_active_mask(train_ds, device)
    pos_weight_tensor = compute_auto_pos_weight(train_ds, args.pos_weight, device) if args.loss == "hurdle_poisson" else None

    if args.model == "minimal":
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
    elif args.model == "official-r":
        if args.temporal_layers != args.spatial_layers:
            LOG.warning(
                "official-r usa um unico num_layers para bloco temporal e espacial; "
                "usando temporal_layers=%d e ignorando spatial_layers=%d",
                args.temporal_layers,
                args.spatial_layers,
            )
        model = STConvS2SOfficialRGrid(
            in_channels=45,
            hidden_channels=args.hidden_channels,
            out_channels=1,
            t_in=28,
            t_out=28,
            num_layers=args.temporal_layers,
            kernel_size=3,
            dropout=args.dropout,
            height=args.grid_h,
            width=args.grid_w,
        ).to(device)
    elif args.model == "official-r-hurdle":
        if args.loss != "hurdle_poisson":
            LOG.warning("official-r-hurdle normalmente deve usar --loss hurdle_poisson; recebido %s", args.loss)
        if args.temporal_layers != args.spatial_layers:
            LOG.warning(
                "official-r-hurdle usa um unico num_layers para bloco temporal e espacial; "
                "usando temporal_layers=%d e ignorando spatial_layers=%d",
                args.temporal_layers,
                args.spatial_layers,
            )
        model = STConvS2SOfficialRHurdleGrid(
            in_channels=45,
            hidden_channels=args.hidden_channels,
            t_in=28,
            t_out=28,
            num_layers=args.temporal_layers,
            kernel_size=3,
            dropout=args.dropout,
            height=args.grid_h,
            width=args.grid_w,
        ).to(device)
    elif args.model == "upstream-r":
        if args.temporal_layers != args.spatial_layers:
            LOG.warning(
                "upstream-r usa um unico num_layers para bloco temporal e espacial; "
                "usando temporal_layers=%d e ignorando spatial_layers=%d",
                args.temporal_layers,
                args.spatial_layers,
            )
        model = STConvS2SUpstreamRGrid(
            in_channels=45,
            hidden_channels=args.hidden_channels,
            out_channels=1,
            t_in=28,
            t_out=28,
            num_layers=args.temporal_layers,
            kernel_size=3,
            dropout=args.dropout,
            device=str(device),
            height=args.grid_h,
            width=args.grid_w,
        ).to(device)
    elif args.model == "upstream-r-hurdle":
        if args.loss != "hurdle_poisson":
            LOG.warning("upstream-r-hurdle normalmente deve usar --loss hurdle_poisson; recebido %s", args.loss)
        if args.temporal_layers != args.spatial_layers:
            LOG.warning(
                "upstream-r-hurdle usa um unico num_layers para bloco temporal e espacial; "
                "usando temporal_layers=%d e ignorando spatial_layers=%d",
                args.temporal_layers,
                args.spatial_layers,
            )
        model = STConvS2SUpstreamRHurdleGrid(
            in_channels=45,
            hidden_channels=args.hidden_channels,
            t_in=28,
            t_out=28,
            num_layers=args.temporal_layers,
            kernel_size=3,
            dropout=args.dropout,
            device=str(device),
            height=args.grid_h,
            width=args.grid_w,
        ).to(device)
    else:
        raise ValueError(f"modelo desconhecido: {args.model}")
    LOG.info("Modelo: %s", model.__class__.__name__)
    LOG.info("Parametros treinaveis: %d", count_parameters(model))

    first_batch = next(iter(train_loader))
    x0, y0 = to_model_tensors(first_batch, device)
    with torch.no_grad():
        p0 = model(x0)
    if isinstance(p0, dict):
        LOG.info(
            "Contrato modelo: X %s -> occ %s, log_lambda %s; target %s",
            tuple(x0.shape),
            tuple(p0["occ_logits"].shape),
            tuple(p0["log_lambda"].shape),
            tuple(y0.shape),
        )
    else:
        LOG.info("Contrato modelo: X %s -> pred %s; target %s", tuple(x0.shape), tuple(p0.shape), tuple(y0.shape))
    assert_prediction_shape(p0, y0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_rows: list[dict[str, float]] = []
    best_metric = float("inf")
    best_loss = float("inf")
    best_metric_epoch = -1
    best_loss_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.loss,
            active_mask,
            args.max_train_batches,
            args.occurrence_weight,
            args.count_weight,
            args.occurrence_threshold,
            pos_weight_tensor,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.loss,
            active_mask,
            args.max_val_batches,
            prefix="val",
            occurrence_weight=args.occurrence_weight,
            count_weight=args.count_weight,
            occurrence_threshold=args.occurrence_threshold,
            pos_weight_tensor=pos_weight_tensor,
        )
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        metrics_rows.append(row)
        write_metrics_csv(outdir / "metrics.csv", metrics_rows)

        LOG.info(
            "epoch=%03d train_loss=%.6f val_loss=%.6f val_mae_active=%.4f "
            "val_rounded_mae_active=%.4f val_mae_h28=%.4f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_mae_all28_activecells"],
            row["val_rounded_mae_all28_activecells"],
            row["val_mae_h28_allcells"],
        )

        if args.checkpoint_metric not in row:
            raise KeyError(f"checkpoint_metric ausente em metrics: {args.checkpoint_metric}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_metric_value": float(row[args.checkpoint_metric]),
            "val_loss": float(row["val_loss"]),
        }
        torch.save(checkpoint, outdir / "last.pt")
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            best_loss_epoch = epoch
            torch.save(checkpoint, outdir / "best_loss.pt")
        if row[args.checkpoint_metric] < best_metric:
            best_metric = row[args.checkpoint_metric]
            best_metric_epoch = epoch
            torch.save(checkpoint, outdir / "best.pt")

    LOG.info(
        "Concluido. best_metric=%s %.6f epoch=%d; best_val_loss=%.6f epoch=%d",
        args.checkpoint_metric,
        best_metric,
        best_metric_epoch,
        best_loss,
        best_loss_epoch,
    )
    LOG.info("Artefatos: %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

