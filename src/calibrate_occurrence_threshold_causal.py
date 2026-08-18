#!/usr/bin/env python3
"""Select Bernoulli–Poisson occurrence threshold on causal-grid validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arboseer" / "src"))
from data.causal_epidemiological_dataset import CausalEpidemiologicalTensorDataset  # noqa: E402
from eval_stconv_s2s_e1 import build_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_path(value: str | None, fallback: Path) -> Path:
    path = Path(value) if value else fallback
    return path if path.is_absolute() else ROOT / path


def horizon_metrics(estimate: np.ndarray, target: np.ndarray, observation: np.ndarray) -> dict:
    metrics = {}
    for index in range(4):
        error = estimate[:, index][:, observation] - target[:, index][:, observation]
        horizon = index + 1
        metrics[f"mae_h{horizon}"] = float(np.abs(error).mean())
        metrics[f"rmse_h{horizon}"] = float(np.sqrt(np.square(error).mean()))
        metrics[f"bias_h{horizon}"] = float(error.mean())
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = checkpoint["config"]
    if cfg.get("dataset_contract") != "causal-grid" or cfg.get("loss") != "hurdle_poisson":
        raise ValueError("É necessário checkpoint Bernoulli–Poisson do contrato causal-grid")
    if int(cfg.get("output_steps", 0)) != 4:
        raise ValueError("Calibração V2 desta etapa exige saída semanal h1–h4")
    checkpoint_metric = checkpoint.get("checkpoint_metric") or cfg.get("checkpoint_metric")
    if checkpoint_metric != "val_mae_h1_observedcells":
        raise RuntimeError(
            "Critério do checkpoint incompatível: esperado val_mae_h1_observedcells, "
            f"recebido {checkpoint_metric!r}"
        )
    epi = configured_path(cfg.get("epidemiology_dir"), ROOT / "missing_epidemiology_dir")
    calibration = configured_path(cfg.get("calibration_dir"), epi)
    cases_path, dates_path = epi / "cases.npy", epi / "dates.npy"
    dynamic_indices = configured_path(cfg.get("dynamic_cell_indices_path"), epi / "dynamic_cell_indices.npy")
    dynamic_features = configured_path(cfg.get("dynamic_features_path"), epi / "missing_dynamic_features.npy")
    dynamic_channels = configured_path(cfg.get("dynamic_channels_path"), epi / "missing_dynamic_channels.json")
    required = [cases_path, dates_path, epi / "masks.npz",
                calibration / "validation_anchor_indices.npy",
                calibration / f"scaler_epi_lag{cfg['lag_count']}.npz", dynamic_indices,
                dynamic_features, dynamic_channels, epi / "manifest.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Contrato tensorial V2 incompleto:\n" + "\n".join(missing))
    manifest = json.loads((epi / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("purpose") != "development" or manifest.get("output_steps") != 4
            or max(manifest.get("dates", [""])) >= "2023-01-01"):
        raise RuntimeError("Firewall violado: calibrador aceita somente desenvolvimento até 2022")
    dates = np.load(dates_path)
    anchors = np.load(calibration / "validation_anchor_indices.npy").astype(np.int64)
    target_indices = anchors[:, None] + np.arange(1, 5, dtype=np.int64)[None, :]
    if target_indices.size == 0 or target_indices.max() >= len(dates):
        raise RuntimeError("Âncoras de validação vazias ou fora do eixo temporal")
    target_years = dates[target_indices].astype("datetime64[Y]").astype(int) + 1970
    if set(np.unique(target_years)) != {2021}:
        raise RuntimeError(f"Firewall violado: validation contém anos {sorted(np.unique(target_years).tolist())}")
    ds = CausalEpidemiologicalTensorDataset(
        cases_path=cases_path, dates_path=dates_path,
        masks_path=epi / "masks.npz", anchor_indices=calibration / "validation_anchor_indices.npy",
        input_steps=cfg["input_steps"], output_steps=cfg["output_steps"], lag_count=cfg["lag_count"],
        dynamic_features_path=dynamic_features,
        dynamic_channels_path=dynamic_channels,
        dynamic_cell_indices_path=dynamic_indices,
        epidemiological_scaler_path=calibration / f"scaler_epi_lag{cfg['lag_count']}.npz",
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probs, rates, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device=device, dtype=torch.float32).permute(0, 2, 1, 3, 4)
            pred = model(x)
            probs.append(torch.sigmoid(pred["occ_logits"]).squeeze(1).cpu().numpy())
            rates.append(torch.exp(pred["log_lambda"]).squeeze(1).cpu().numpy())
            targets.append(batch["Y"].numpy())
    probability, rate, target = map(lambda chunks: np.concatenate(chunks), (probs, rates, targets))
    if probability.shape != target.shape or rate.shape != target.shape or probability.shape[1] != 4:
        raise RuntimeError(
            f"Contrato de saída inválido: probability={probability.shape}, rate={rate.shape}, target={target.shape}"
        )
    if not (np.isfinite(probability).all() and np.isfinite(rate).all() and np.isfinite(target).all()):
        raise RuntimeError("Predições ou alvos não finitos durante calibração")
    if not np.any(ds.observation_mask):
        raise RuntimeError("Máscara de observação vazia")
    thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(v <= 0 or v >= 1 for v in thresholds):
        raise ValueError("Thresholds devem ser únicos e pertencer ao intervalo aberto (0,1)")
    rows = []
    for threshold in thresholds:
        estimate = np.where(probability >= threshold, rate, 0.0)
        rows.append({"threshold": threshold, **horizon_metrics(estimate, target, ds.observation_mask)})
    best = min(rows, key=lambda row: (row["mae_h1"], row["rmse_h1"], row["threshold"]))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    checkpoint_path = Path(args.checkpoint).resolve()
    payload = {**best, "split": "validation", "validation_year": 2021,
               "selection_metric": "mae_h1", "tie_break": ["rmse_h1", "lower_threshold"],
               "secondary_metrics": [f"{metric}_h{h}" for h in range(1, 5)
                                     for metric in ("mae", "rmse", "bias") if not (metric == "mae" and h == 1)],
               "checkpoint_metric": checkpoint_metric, "checkpoint": str(checkpoint_path),
               "checkpoint_metric_value": checkpoint.get("checkpoint_metric_value"),
               "checkpoint_epoch": checkpoint.get("epoch"),
               "checkpoint_sha256": sha256(checkpoint_path), "samples": len(ds),
               "metric_scope": "continuous predicted counts on observation-mask cells",
               "threshold_candidates": thresholds,
               "tensor_manifest": str((epi / "manifest.json").resolve()),
               "dynamic_cell_indices_path": str(dynamic_indices.resolve()),
               "confirmation_2022_loaded": False, "final_2023_loaded": False}
    (out / "best_threshold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
