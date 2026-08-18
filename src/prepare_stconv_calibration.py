#!/usr/bin/env python3
"""Prepare the pre-specified STConv calibration plan without fitting models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


LAG_CANDIDATES = (4, 6, 8)
LOOKBACK_WEEKS = 4
LOSS_CANDIDATES = (
    ("poisson", "official-r", "poisson_nll_log"),
    ("hurdle", "official-r-hurdle", "hurdle_poisson"),
)
CALIBRATION_SEED = 987


def target_years(dates: np.ndarray) -> np.ndarray:
    return pd.DatetimeIndex(dates).isocalendar().year.to_numpy(dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epidemiology-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-allowed-year", type=int, default=2022)
    args = parser.parse_args()

    source = Path(args.epidemiology_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_2023_loaded") is not False:
        raise RuntimeError("Firewall violado: manifesto não comprova exclusão de 2023")
    dates = np.load(source / "week_dates.npy")
    years = target_years(dates)
    if int(years.max()) > args.max_allowed_year:
        raise RuntimeError(
            f"Firewall violado: série contém ano {int(years.max())}, "
            f"máximo permitido {args.max_allowed_year}"
        )
    if set(np.unique(years)) - set(range(2014, 2023)):
        raise RuntimeError(f"Anos inesperados na base: {sorted(set(years))}")

    # Âncora a prevê o alvo a+1. O mesmo conjunto de alvos é usado em todas as
    # alternativas. O lookback foi fixado a priori em quatro semanas.
    target_indices = np.arange(len(dates), dtype=np.int64)
    # O primeiro frame da entrada também precisa de todos os lags. Para um
    # alvo t: primeiro frame = t-lookback e ele requer lag_count-1 semanas.
    minimum_target = LOOKBACK_WEEKS + max(LAG_CANDIDATES) - 1
    target_indices = target_indices[target_indices >= minimum_target]
    split_years = {
        "train": set(range(2014, 2021)),
        "validation": {2021},
        "confirmation": {2022},
    }
    anchor_summary = {}
    for split, allowed in split_years.items():
        selected_targets = target_indices[np.isin(years[target_indices], list(allowed))]
        anchors = selected_targets - 1
        np.save(out / f"{split}_anchor_indices.npy", anchors)
        anchor_summary[split] = {
            "samples": int(len(anchors)),
            "first_target": str(dates[selected_targets[0]]) if len(anchors) else None,
            "last_target": str(dates[selected_targets[-1]]) if len(anchors) else None,
        }

    rows = []
    run = 0
    for lag_count in LAG_CANDIDATES:
        for loss_label, model, loss in LOSS_CANDIDATES:
            run += 1
            rows.append({
                    "run_id": f"A{run:02d}",
                    "phase": "A_data_and_loss",
                    "lag_count": lag_count,
                    "epidemiological_channels": 8 + lag_count,
                    "lookback_weeks": LOOKBACK_WEEKS,
                    "horizon_weeks": 1,
                    "loss_candidate": loss_label,
                    "model": model,
                    "loss": loss,
                    "hidden_channels": 4,
                    "temporal_layers": 1,
                    "spatial_layers": 1,
                    "dropout": 0.0,
                    "seed": CALIBRATION_SEED,
                    "selection_split": "validation_2021",
                    "primary_metric": "MAE_h1_observation_mask",
                    "secondary_metrics": "RMSE_h1;bias_h1;Poisson_deviance_h1;runtime",
                    "status": "PLANNED_NOT_RUN",
            })
    with (out / "phase_a_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    protocol = {
        "status": "PRE_SPECIFIED_NOT_RUN",
        "source": str(source),
        "source_manifest": manifest,
        "firewall": {"max_allowed_year": args.max_allowed_year, "year_2023_loaded": False},
        "common_target_support": anchor_summary,
        "minimum_target_index": int(minimum_target),
        "phase_a_runs": len(rows),
        "calibration_seed": CALIBRATION_SEED,
        "lookback_weeks_fixed": LOOKBACK_WEEKS,
        "tie_break": ["lower_validation_MAE", "lower_validation_RMSE", "fewer_lags"],
        "confirmation_rule": (
            "Após congelar D024, D025, D034 e D035 com validação de 2021, "
            "executar somente a configuração escolhida em 2022; não retunar com 2022."
        ),
    }
    (out / "protocol_manifest.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"phase_a_runs": len(rows), "anchors": anchor_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
