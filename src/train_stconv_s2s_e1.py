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

from data.spatiotemporal_tensor_dataset import SpatiotemporalTensorDataset  # noqa: E402
from data.causal_epidemiological_dataset import CausalEpidemiologicalTensorDataset  # noqa: E402
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
    training_mode: str
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
    input_steps: int
    output_steps: int
    in_channels: int
    mask_policy: str
    early_stopping_patience: int
    dataset_contract: str
    epidemiology_dir: str
    calibration_dir: str
    lag_count: int
    dynamic_features_path: str
    dynamic_channels_path: str
    dynamic_cell_indices_path: str


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


def make_active_mask(train_dataset, device: torch.device) -> torch.Tensor:
    if isinstance(train_dataset, CausalEpidemiologicalTensorDataset):
        raise ValueError(
            "mask_policy=active é proibida no contrato causal; use observation para "
            "não definir supervisão pela ocorrência de casos"
        )
    active = (train_dataset._targets.sum(axis=0) > 0).astype(np.float32)  # noqa: SLF001
    mask = torch.from_numpy(active)[None, None, None].to(device=device)
    LOG.info("Mascara ativa: %d/%d celulas", int(active.sum()), active.size)
    return mask


def make_observation_mask(train_dataset, device: torch.device) -> torch.Tensor:
    """All cells containing an experimental CNES unit, regardless of territory mask."""
    if getattr(train_dataset, "observation_mask", None) is not None:
        valid = np.asarray(train_dataset.observation_mask, dtype=np.float32)
    else:
        if "CNES_MASK" not in train_dataset.channels:
            raise ValueError("Canal CNES_MASK necessário para observation mask")
        cnes = train_dataset._feats[:, train_dataset.channels.index("CNES_MASK")] > 0.5  # noqa: SLF001
        valid = cnes.all(axis=0).astype(np.float32)
    mask = torch.from_numpy(valid)[None, None, None].to(device=device)
    LOG.info("Mascara de observacao CNES: %d/%d celulas", int(valid.sum()), valid.size)
    return mask


def masked_mean(values: torch.Tensor, spatial_mask: torch.Tensor) -> torch.Tensor:
    mask = spatial_mask.expand_as(values).to(dtype=values.dtype)
    selected = torch.where(mask.bool(), values, torch.zeros_like(values))
    return selected.sum() / mask.sum().clamp_min(1.0)


def compute_loss(
    pred,
    target: torch.Tensor,
    loss_name: str,
    occurrence_weight: float = 1.0,
    count_weight: float = 1.0,
    pos_weight_tensor: torch.Tensor | None = None,
    observation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if observation_mask is None:
        observation_mask = torch.ones((1, 1, 1, target.shape[-2], target.shape[-1]), device=target.device)
    if loss_name == "hurdle_poisson":
        if not isinstance(pred, dict):
            raise TypeError("hurdle_poisson espera pred dict com occ_logits e log_lambda")
        occ_target = (target > 0).to(dtype=target.dtype)
        bce = F.binary_cross_entropy_with_logits(
            pred["occ_logits"],
            occ_target,
            pos_weight=pos_weight_tensor,
            reduction="none",
        )
        bce = masked_mean(bce, observation_mask)
        positive_mask = (target > 0) & observation_mask.expand_as(target).bool()
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
        element = F.poisson_nll_loss(pred, target, log_input=True, full=False, reduction="none")
        return masked_mean(element, observation_mask)
    if loss_name == "mse_log1p":
        target_log = torch.log1p(target)
        return masked_mean(F.mse_loss(pred, target_log, reduction="none"), observation_mask)
    if loss_name == "mae_log1p":
        target_log = torch.log1p(target)
        return masked_mean(F.l1_loss(pred, target_log, reduction="none"), observation_mask)
    if loss_name == "mse_raw":
        return masked_mean(F.mse_loss(pred, target, reduction="none"), observation_mask)
    if loss_name == "mae_raw":
        return masked_mean(F.l1_loss(pred, target, reduction="none"), observation_mask)
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
        "sum_signed_all": 0.0,
        "sum_abs_all": 0.0,
        "sum_sq_all": 0.0,
        "n_all": 0.0,
        "sum_signed_active": 0.0,
        "sum_abs_active": 0.0,
        "sum_sq_active": 0.0,
        "n_active": 0.0,
        "horizon_abs": {h: 0.0 for h in EVAL_HORIZONS},
        "horizon_n": {h: 0.0 for h in EVAL_HORIZONS},
        "horizon_abs_observed": {h: 0.0 for h in EVAL_HORIZONS},
        "horizon_n_observed": {h: 0.0 for h in EVAL_HORIZONS},
    }


def update_metrics(acc: dict, pred_counts: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> None:
    diff = pred_counts - target
    abs_diff = diff.abs()
    sq_diff = diff.square()

    acc["sum_signed_all"] += float(diff.sum().detach().cpu())
    acc["sum_abs_all"] += float(abs_diff.sum().detach().cpu())
    acc["sum_sq_all"] += float(sq_diff.sum().detach().cpu())
    acc["n_all"] += float(abs_diff.numel())

    active = active_mask.expand_as(abs_diff).bool()
    acc["sum_signed_active"] += float(torch.where(active, diff, 0.0).sum().detach().cpu())
    acc["sum_abs_active"] += float(torch.where(active, abs_diff, 0.0).sum().detach().cpu())
    acc["sum_sq_active"] += float(torch.where(active, sq_diff, 0.0).sum().detach().cpu())
    acc["n_active"] += float(active.sum().detach().cpu())

    for h in EVAL_HORIZONS:
        idx = h - 1
        h_abs = abs_diff[:, :, idx]
        acc["horizon_abs"][h] += float(h_abs.sum().detach().cpu())
        acc["horizon_n"][h] += float(h_abs.numel())
        h_mask = active_mask.expand_as(abs_diff)[:, :, idx].bool()
        acc["horizon_abs_observed"][h] += float(
            torch.where(h_mask, h_abs, 0.0).sum().detach().cpu()
        )
        acc["horizon_n_observed"][h] += float(h_mask.sum().detach().cpu())


def finalize_metrics(acc: dict, prefix: str) -> dict[str, float]:
    out = {
        f"{prefix}_bias_all{EXPECTED_T_OUT}_allcells": acc["sum_signed_all"] / max(acc["n_all"], 1.0),
        f"{prefix}_mae_all{EXPECTED_T_OUT}_allcells": acc["sum_abs_all"] / max(acc["n_all"], 1.0),
        f"{prefix}_rmse_all{EXPECTED_T_OUT}_allcells": math.sqrt(acc["sum_sq_all"] / max(acc["n_all"], 1.0)),
        f"{prefix}_bias_all{EXPECTED_T_OUT}_activecells": acc["sum_signed_active"] / max(acc["n_active"], 1.0),
        f"{prefix}_mae_all{EXPECTED_T_OUT}_activecells": acc["sum_abs_active"] / max(acc["n_active"], 1.0),
        f"{prefix}_rmse_all{EXPECTED_T_OUT}_activecells": math.sqrt(acc["sum_sq_active"] / max(acc["n_active"], 1.0)),
        # Nomes explícitos usados pelo contrato atual; aliases activecells são
        # preservados acima para leitura de artefatos legados.
        f"{prefix}_bias_all{EXPECTED_T_OUT}_observedcells": acc["sum_signed_active"] / max(acc["n_active"], 1.0),
        f"{prefix}_mae_all{EXPECTED_T_OUT}_observedcells": acc["sum_abs_active"] / max(acc["n_active"], 1.0),
        f"{prefix}_rmse_all{EXPECTED_T_OUT}_observedcells": math.sqrt(acc["sum_sq_active"] / max(acc["n_active"], 1.0)),
    }
    for h in EVAL_HORIZONS:
        out[f"{prefix}_mae_h{h}_allcells"] = acc["horizon_abs"][h] / max(acc["horizon_n"][h], 1.0)
        out[f"{prefix}_mae_h{h}_observedcells"] = (
            acc["horizon_abs_observed"][h] / max(acc["horizon_n_observed"][h], 1.0)
        )
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


def compute_auto_pos_weight(
    train_dataset: SpatiotemporalTensorDataset,
    requested: float,
    device: torch.device,
    observation_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if requested < 0:
        return None
    if requested > 0:
        value = float(requested)
    else:
        y = train_dataset._targets  # noqa: SLF001
        if observation_mask is None:
            valid = np.ones_like(y, dtype=bool)
        else:
            spatial = observation_mask.squeeze().detach().cpu().numpy().astype(bool)
            valid = np.broadcast_to(spatial, y.shape)
        positives = float(((y > 0) & valid).sum())
        total = float(valid.sum())
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
    observation_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    model.train()
    acc_continuous = metric_accumulator()
    acc_rounded = metric_accumulator()
    losses: list[float] = []
    for batch in maybe_limited(loader, max_batches):
        x, target = to_model_tensors(batch, device)
        pred = model(x)
        assert_prediction_shape(pred, target)

        loss = compute_loss(
            pred, target, loss_name, occurrence_weight, count_weight,
            pos_weight_tensor, observation_mask,
        )
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
    observation_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    acc_continuous = metric_accumulator()
    acc_rounded = metric_accumulator()
    losses: list[float] = []
    for batch in maybe_limited(loader, max_batches):
        x, target = to_model_tensors(batch, device)
        pred = model(x)
        assert_prediction_shape(pred, target)
        loss = compute_loss(
            pred, target, loss_name, occurrence_weight, count_weight,
            pos_weight_tensor, observation_mask,
        )
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


def build_loaders(args: argparse.Namespace):
    if args.dataset_contract == "causal-grid":
        epidemiology = Path(args.epidemiology_dir)
        calibration = Path(args.calibration_dir)
        cases_path = epidemiology / "cases.npy"
        dates_path = epidemiology / "dates.npy"
        if not cases_path.exists():
            cases_path = epidemiology / "cases_weekly.npy"
        if not dates_path.exists():
            dates_path = epidemiology / "week_dates.npy"
        common = dict(
            cases_path=cases_path,
            dates_path=dates_path,
            masks_path=epidemiology / "masks.npz",
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            lag_count=args.lag_count,
            dynamic_features_path=args.dynamic_features_path or None,
            dynamic_channels_path=args.dynamic_channels_path or None,
            dynamic_cell_indices_path=args.dynamic_cell_indices_path or None,
            epidemiological_scaler_path=calibration / f"scaler_epi_lag{args.lag_count}.npz",
        )
        train_ds = CausalEpidemiologicalTensorDataset(
            anchor_indices=calibration / "train_anchor_indices.npy", **common
        )
        val_ds = (None if args.training_mode == "final-refit" else
                  CausalEpidemiologicalTensorDataset(
                      anchor_indices=calibration / "validation_anchor_indices.npy", **common))
    else:
        train_ds = SpatiotemporalTensorDataset(
            dataset_dir=args.dataset_dir, split="train",
            input_steps=args.input_steps, output_steps=args.output_steps,
        )
        val_ds = SpatiotemporalTensorDataset(
            dataset_dir=args.dataset_dir, split="val",
            input_steps=args.input_steps, output_steps=args.output_steps,
        )

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
    val_loader = (None if val_data is None else DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()))
    return train_loader, val_loader, train_ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trainer STConvS2S configurável para RJ ou Natal"
    )
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
    parser.add_argument("--training-mode", choices=["selection", "final-refit"], default="selection")
    parser.add_argument("--frozen-protocol", default="")
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
    parser.add_argument(
        "--dataset-contract", choices=["legacy", "causal-grid"], default="legacy",
        help="legacy lê arrays por split; causal-grid gera canais epidemiológicos sob demanda",
    )
    parser.add_argument(
        "--epidemiology-dir",
        default=str(PROJECT_ROOT / "arboseer" / "data" / "processed" / "epidemiology" / "RJ_STATE_2km_WEEKLY_2014_2022"),
    )
    parser.add_argument(
        "--calibration-dir",
        default=str(PROJECT_ROOT / "docs" / "calibracao_cap4" / "stconv_protocol"),
    )
    parser.add_argument("--lag-count", type=int, choices=[4, 6, 8], default=6)
    parser.add_argument("--dynamic-features-path", default="")
    parser.add_argument("--dynamic-channels-path", default="")
    parser.add_argument(
        "--dynamic-cell-indices-path", default="",
        help="Mapa (N,2) para canais dinâmicos compactos (T,C,N)",
    )
    parser.add_argument(
        "--input-steps",
        type=int,
        default=28,
        help="Comprimento da janela de entrada na unidade temporal do dataset",
    )
    parser.add_argument(
        "--output-steps",
        type=int,
        default=28,
        help="Número de passos futuros previstos",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overfit-batches", type=int, default=0, help=">0 usa poucos batches para sanity/overfit")
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 = sem limite")
    parser.add_argument(
        "--checkpoint-metric",
        default="",
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
    parser.add_argument(
        "--mask-policy",
        choices=["active", "observation"],
        default="observation",
        help="active=legado; observation=todas as células CNES na loss e métricas",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="0 desativa; valores positivos monitoram val_loss",
    )
    return parser.parse_args()


def main() -> int:
    global EXPECTED_T_IN, EXPECTED_T_OUT, EXPECTED_CHANNELS, EVAL_HORIZONS
    args = parse_args()
    if args.training_mode == "final-refit":
        if args.dataset_contract != "causal-grid" or args.overfit_batches:
            raise RuntimeError("Reajuste final exige causal-grid e proíbe overfit-batches")
        if not args.frozen_protocol:
            raise RuntimeError("Reajuste final bloqueado: informe --frozen-protocol")
        frozen = json.loads(Path(args.frozen_protocol).read_text(encoding="utf-8"))
        if frozen.get("status") != "FROZEN":
            raise RuntimeError("Reajuste final bloqueado: protocolo ainda não está FROZEN")
        if args.early_stopping_patience != 0:
            raise RuntimeError("Reajuste final não pode usar early stopping; use épocas congeladas")
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
    if args.dataset_contract == "causal-grid":
        LOG.info("Contrato causal: epidemiologia=%s calibracao=%s", args.epidemiology_dir, args.calibration_dir)
    else:
        LOG.info("Dataset dir: %s", args.dataset_dir)
    device = torch.device(args.device)

    train_loader, val_loader, train_ds = build_loaders(args)
    args.in_channels = int(train_ds.in_channels)
    EXPECTED_T_IN = int(args.input_steps)
    EXPECTED_T_OUT = int(args.output_steps)
    EXPECTED_CHANNELS = int(args.in_channels)
    canonical_horizons = (
        list(range(1, EXPECTED_T_OUT + 1))
        if EXPECTED_T_OUT <= 4
        else [1, 7, 14, 21, 28]
    )
    EVAL_HORIZONS = [h for h in canonical_horizons if h <= EXPECTED_T_OUT]
    if EXPECTED_T_OUT not in EVAL_HORIZONS:
        EVAL_HORIZONS.append(EXPECTED_T_OUT)
    if not args.checkpoint_metric:
        args.checkpoint_metric = "val_mae_h1_observedcells"
    grid_h, grid_w = train_ds.grid_shape
    args.grid_h = int(grid_h)
    args.grid_w = int(grid_w)
    LOG.info("Grid dataset: %dx%d", args.grid_h, args.grid_w)
    LOG.info(
        "Contrato tensorial: entrada=(T=%d,C=%d), saída=(T=%d), horizontes=%s",
        EXPECTED_T_IN,
        EXPECTED_CHANNELS,
        EXPECTED_T_OUT,
        EVAL_HORIZONS,
    )

    cfg = RunConfig(**{k: getattr(args, k) for k in RunConfig.__annotations__})
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    if args.mask_policy == "observation":
        active_mask = make_observation_mask(train_ds, device)
        observation_mask = active_mask
    else:
        active_mask = make_active_mask(train_ds, device)
        observation_mask = None
    pos_weight_tensor = (
        compute_auto_pos_weight(train_ds, args.pos_weight, device, observation_mask)
        if args.loss == "hurdle_poisson" else None
    )

    if args.model == "minimal":
        model = STConvS2SGrid(
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=1,
            t_in=args.input_steps,
            t_out=args.output_steps,
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
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=1,
            t_in=args.input_steps,
            t_out=args.output_steps,
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
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            t_in=args.input_steps,
            t_out=args.output_steps,
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
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=1,
            t_in=args.input_steps,
            t_out=args.output_steps,
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
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            t_in=args.input_steps,
            t_out=args.output_steps,
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
    epochs_without_loss_improvement = 0

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
            observation_mask,
        )
        val_metrics = {} if val_loader is None else evaluate(
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
            observation_mask=observation_mask,
        )
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        metrics_rows.append(row)
        write_metrics_csv(outdir / "metrics.csv", metrics_rows)

        if val_loader is not None:
            LOG.info(
            "epoch=%03d train_loss=%.6f val_loss=%.6f val_mae_observed=%.4f "
            "val_bias_observed=%.4f val_rounded_mae_observed=%.4f "
            "val_rounded_bias_observed=%.4f val_mae_h1_observed=%.4f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row[f"val_mae_all{EXPECTED_T_OUT}_observedcells"],
            row[f"val_bias_all{EXPECTED_T_OUT}_observedcells"],
            row[f"val_rounded_mae_all{EXPECTED_T_OUT}_observedcells"],
            row[f"val_rounded_bias_all{EXPECTED_T_OUT}_observedcells"],
            row["val_mae_h1_observedcells"],
        )

        if val_loader is None:
            checkpoint = {"model_state_dict": model.state_dict(), "config": asdict(cfg),
                          "epoch": epoch, "training_mode": "final-refit"}
            torch.save(checkpoint, outdir / "last.pt")
            continue
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
            epochs_without_loss_improvement = 0
            torch.save(checkpoint, outdir / "best_loss.pt")
        else:
            epochs_without_loss_improvement += 1
        if row[args.checkpoint_metric] < best_metric:
            best_metric = row[args.checkpoint_metric]
            best_metric_epoch = epoch
            torch.save(checkpoint, outdir / "best.pt")

        if (
            args.early_stopping_patience > 0
            and epochs_without_loss_improvement >= args.early_stopping_patience
        ):
            LOG.info(
                "Early stopping na epoca %d: val_loss sem melhora por %d epocas",
                epoch,
                epochs_without_loss_improvement,
            )
            break

    if val_loader is None:
        torch.save(checkpoint, outdir / "final.pt")
        LOG.info("Reajuste final concluído em %d épocas congeladas; sem validação/early stopping", args.epochs)
    else:
        LOG.info(
            "Concluido. best_metric=%s %.6f epoch=%d; best_val_loss=%.6f epoch=%d",
            args.checkpoint_metric, best_metric, best_metric_epoch, best_loss, best_loss_epoch)
    LOG.info("Artefatos: %s", outdir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        LOG.exception("Falha não tratada durante o treino")
        raise

