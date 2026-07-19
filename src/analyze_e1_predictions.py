#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics for E1/E2 STConvS2S prediction archives.

Input archive contract:
  pred_counts, pred_rounded, target: (N, 28, H, W)
  anchor_dates: (N,)
  horizons: (28,)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PREDICTIONS = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "RJ_E1_T1_clean_hurdle_hparam_train_test_20260704_183024"
    / "test_selected_G_pos10_occ100_best_thr087"
    / "predictions_test.npz"
)


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x))) if x.size else 0.0


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def analyze(predictions_path: Path, output_dir: Path, grid_lat_min: float, grid_lat_max: float, grid_lon_min: float, grid_lon_max: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(predictions_path)
    pred_counts = data["pred_counts"].astype(np.float32)
    pred = data["pred_rounded"].astype(np.float32)
    target = data["target"].astype(np.float32)
    anchor_dates = data["anchor_dates"].astype(str)
    horizons = data["horizons"].astype(int)

    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    n, t_out, h, w = pred.shape
    lats = np.round(np.linspace(grid_lat_min, grid_lat_max, h), 4)
    lons = np.round(np.linspace(grid_lon_min, grid_lon_max, w), 4)

    diff = pred - target
    abs_diff = np.abs(diff)
    sq_diff = diff * diff
    active_cells = target.sum(axis=(0, 1)) > 0
    active_den = max(int(active_cells.sum()) * n * t_out, 1)

    summary = {
        "predictions_path": str(predictions_path),
        "shape": list(pred.shape),
        "n_samples": int(n),
        "grid_shape": [int(h), int(w)],
        "active_cells": int(active_cells.sum()),
        "total_cells": int(h * w),
        "target_total": float(target.sum()),
        "pred_total_rounded": float(pred.sum()),
        "pred_total_continuous": float(pred_counts.sum()),
        "target_nonzero": int((target > 0).sum()),
        "pred_nonzero_rounded": int((pred > 0).sum()),
        "target_max": float(target.max()),
        "pred_max_rounded": float(pred.max()),
        "mae_all": mae(diff),
        "rmse_all": rmse(diff),
        "mae_active": float((abs_diff * active_cells[None, None]).sum() / active_den),
        "rmse_active": float(np.sqrt((sq_diff * active_cells[None, None]).sum() / active_den)),
    }

    occurrence_target = target > 0
    occurrence_pred = pred > 0
    tp = int((occurrence_target & occurrence_pred).sum())
    fp = int((~occurrence_target & occurrence_pred).sum())
    fn = int((occurrence_target & ~occurrence_pred).sum())
    tn = int((~occurrence_target & ~occurrence_pred).sum())
    summary.update(
        {
            "occurrence_tp": tp,
            "occurrence_fp": fp,
            "occurrence_fn": fn,
            "occurrence_tn": tn,
            "occurrence_precision": safe_div(tp, tp + fp),
            "occurrence_recall": safe_div(tp, tp + fn),
            "occurrence_f1": safe_div(2 * tp, 2 * tp + fp + fn),
        }
    )

    horizon_rows = []
    for idx, horizon in enumerate(horizons):
        y = target[:, idx]
        p = pred[:, idx]
        d = p - y
        row = {
            "horizon": int(horizon),
            "target_total": float(y.sum()),
            "pred_total": float(p.sum()),
            "target_nonzero": int((y > 0).sum()),
            "pred_nonzero": int((p > 0).sum()),
            "mae_all": mae(d),
            "rmse_all": rmse(d),
            "mae_active": float((np.abs(d) * active_cells).sum() / max(int(active_cells.sum()) * n, 1)),
            "rmse_active": float(np.sqrt((np.square(d) * active_cells).sum() / max(int(active_cells.sum()) * n, 1))),
        }
        horizon_rows.append(row)
    write_csv(output_dir / "error_by_horizon.csv", horizon_rows)

    cell_rows = []
    total_by_cell = target.sum(axis=(0, 1))
    pred_by_cell = pred.sum(axis=(0, 1))
    mae_by_cell = abs_diff.mean(axis=(0, 1))
    rmse_by_cell = np.sqrt(sq_diff.mean(axis=(0, 1)))
    fn_by_cell = (occurrence_target & ~occurrence_pred).sum(axis=(0, 1))
    fp_by_cell = (~occurrence_target & occurrence_pred).sum(axis=(0, 1))
    for i in range(h):
        for j in range(w):
            cell_rows.append(
                {
                    "cell_i": i,
                    "cell_j": j,
                    "lat": float(lats[i]),
                    "lon": float(lons[j]),
                    "active": bool(active_cells[i, j]),
                    "target_total": float(total_by_cell[i, j]),
                    "pred_total": float(pred_by_cell[i, j]),
                    "mae": float(mae_by_cell[i, j]),
                    "rmse": float(rmse_by_cell[i, j]),
                    "false_negatives": int(fn_by_cell[i, j]),
                    "false_positives": int(fp_by_cell[i, j]),
                }
            )
    cell_df = pd.DataFrame(cell_rows)
    cell_df.to_csv(output_dir / "error_by_cell.csv", index=False, encoding="utf-8")

    np.save(output_dir / "error_map_mae.npy", mae_by_cell.astype(np.float32))
    np.save(output_dir / "error_map_rmse.npy", rmse_by_cell.astype(np.float32))

    top_target = cell_df.sort_values("target_total", ascending=False).head(10)
    top_error = cell_df.sort_values("mae", ascending=False).head(10)
    top_fn = cell_df.sort_values("false_negatives", ascending=False).head(10)

    # One row per anchor/horizon for the top cells by target volume.
    ts_rows = []
    for row in top_target.itertuples(index=False):
        i, j = int(row.cell_i), int(row.cell_j)
        for sample_idx, anchor in enumerate(anchor_dates):
            for horizon_idx, horizon in enumerate(horizons):
                ts_rows.append(
                    {
                        "cell_i": i,
                        "cell_j": j,
                        "lat": float(lats[i]),
                        "lon": float(lons[j]),
                        "anchor_date": anchor,
                        "horizon": int(horizon),
                        "target": float(target[sample_idx, horizon_idx, i, j]),
                        "pred_rounded": float(pred[sample_idx, horizon_idx, i, j]),
                        "pred_counts": float(pred_counts[sample_idx, horizon_idx, i, j]),
                    }
                )
    write_csv(output_dir / "top_cells_timeseries_long.csv", ts_rows)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Diagnostico das predicoes E1",
        "",
        f"- Arquivo: `{predictions_path}`",
        f"- Shape: `{tuple(pred.shape)}`",
        f"- Celulas ativas: **{summary['active_cells']} / {summary['total_cells']}**",
        f"- Total real: **{summary['target_total']:.0f}**",
        f"- Total predito arredondado: **{summary['pred_total_rounded']:.0f}**",
        f"- Max real: **{summary['target_max']:.0f}**",
        f"- Max predito arredondado: **{summary['pred_max_rounded']:.0f}**",
        "",
        "## Metricas globais",
        "",
        f"- MAE all: **{summary['mae_all']:.4f}**",
        f"- RMSE all: **{summary['rmse_all']:.4f}**",
        f"- MAE active: **{summary['mae_active']:.4f}**",
        f"- RMSE active: **{summary['rmse_active']:.4f}**",
        "",
        "## Ocorrencia",
        "",
        f"- Precision: **{summary['occurrence_precision']:.4f}**",
        f"- Recall: **{summary['occurrence_recall']:.4f}**",
        f"- F1: **{summary['occurrence_f1']:.4f}**",
        f"- TP/FP/FN/TN: **{tp}/{fp}/{fn}/{tn}**",
        "",
        "## Top 10 celulas por volume real",
        "",
        "| cell | lat | lon | real | predito | MAE | RMSE | FN | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_target.itertuples(index=False):
        lines.append(
            f"| ({int(row.cell_i)}, {int(row.cell_j)}) | {row.lat:.4f} | {row.lon:.4f} | "
            f"{row.target_total:.0f} | {row.pred_total:.0f} | {row.mae:.4f} | {row.rmse:.4f} | "
            f"{int(row.false_negatives)} | {int(row.false_positives)} |"
        )

    lines += [
        "",
        "## Top 10 celulas por MAE",
        "",
        "| cell | lat | lon | real | predito | MAE | RMSE | FN | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_error.itertuples(index=False):
        lines.append(
            f"| ({int(row.cell_i)}, {int(row.cell_j)}) | {row.lat:.4f} | {row.lon:.4f} | "
            f"{row.target_total:.0f} | {row.pred_total:.0f} | {row.mae:.4f} | {row.rmse:.4f} | "
            f"{int(row.false_negatives)} | {int(row.false_positives)} |"
        )

    lines += [
        "",
        "## Arquivos gerados",
        "",
        "- `error_by_horizon.csv`",
        "- `error_by_cell.csv`",
        "- `top_cells_timeseries_long.csv`",
        "- `error_map_mae.npy`",
        "- `error_map_rmse.npy`",
        "- `summary.json`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze STConvS2S prediction archive")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--grid-lat-min", type=float, default=-23.0)
    parser.add_argument("--grid-lat-max", type=float, default=-22.0)
    parser.add_argument("--grid-lon-min", type=float, default=-44.0)
    parser.add_argument("--grid-lon-max", type=float, default=-42.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent / "diagnostics"
    analyze(
        predictions_path=predictions_path,
        output_dir=output_dir,
        grid_lat_min=args.grid_lat_min,
        grid_lat_max=args.grid_lat_max,
        grid_lon_min=args.grid_lon_min,
        grid_lon_max=args.grid_lon_max,
    )
    print(f"Diagnostics written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
