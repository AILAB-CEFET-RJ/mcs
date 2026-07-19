#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare E1 test metrics across STConvS2S and simple baselines.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "arboseer" / "models" / "RJ_E1_T1_clean_comparison"

RUNS = {
    "STConvS2S-R Poisson": PROJECT_ROOT
    / "arboseer"
    / "models"
    / "RJ_E1_T1_clean_stconv_s2s_train_20260704_000548"
    / "eval_test"
    / "metrics_test.json",
    "STConvS2S-R Hurdle": PROJECT_ROOT
    / "arboseer"
    / "models"
    / "RJ_E1_T1_clean_stconv_s2s_train_20260704_013317"
    / "eval_test"
    / "metrics_test.json",
    "STConvS2S-R Hurdle best-loss": PROJECT_ROOT
    / "arboseer"
    / "models"
    / "RJ_E1_T1_clean_stconv_s2s_train_20260704_013317"
    / "eval_test_best_loss"
    / "metrics_test.json",
    "STConvS2S-R Hurdle calibrated": PROJECT_ROOT
    / "arboseer"
    / "models"
    / "RJ_E1_T1_clean_hurdle_hparam_train_test_20260704_183024"
    / "test_selected_G_pos10_occ100_best_thr087"
    / "metrics_test.json",
    "Persistence": PROJECT_ROOT
    / "arboseer"
    / "models"
    / "RJ_E1_T1_clean_baselines"
    / "metrics_test_persistence.json",
    "Moving Average 7d": PROJECT_ROOT
    / "arboseer"
    / "models"
    / "RJ_E1_T1_clean_baselines"
    / "metrics_test_moving_average_7.json",
    "Moving Average 28d": PROJECT_ROOT
    / "arboseer"
    / "models"
    / "RJ_E1_T1_clean_baselines"
    / "metrics_test_moving_average_28.json",
}

SUMMARY_KEYS = [
    "test_rounded_mae_all28_activecells",
    "test_rounded_mae_all28_allcells",
    "test_rounded_rmse_all28_activecells",
    "test_rounded_rmse_all28_allcells",
    "test_rounded_mae_h1_allcells",
    "test_rounded_mae_h7_allcells",
    "test_rounded_mae_h14_allcells",
    "test_rounded_mae_h21_allcells",
    "test_rounded_mae_h28_allcells",
]


def load_metrics() -> dict[str, dict]:
    metrics = {}
    missing = []
    for name, path in RUNS.items():
        if not path.exists():
            missing.append(str(path))
            continue
        metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise FileNotFoundError("Metric files missing:\n" + "\n".join(missing))
    return metrics


def write_csv(metrics: dict[str, dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, values in metrics.items():
        row = {"method": name}
        for key in SUMMARY_KEYS:
            row[key] = values.get(key)
        rows.append(row)

    with open(OUT_DIR / "comparison_test_rounded.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *SUMMARY_KEYS])
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT_DIR / "comparison_test_rounded.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def best_by_metric(metrics: dict[str, dict], key: str) -> tuple[str, float]:
    best_name = min(metrics, key=lambda name: metrics[name][key])
    return best_name, float(metrics[best_name][key])


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_report(metrics: dict[str, dict]) -> None:
    lines = [
        "# Comparacao E1 Test — STConvS2S-R vs Baselines",
        "",
        "Todas as metricas abaixo usam previsoes inteiras nao negativas (`round(clamp(pred, 0))`).",
        "",
        "## Ranking Agregado",
        "",
        "| Metodo | MAE active | MAE all | RMSE active | RMSE all |",
        "|---|---:|---:|---:|---:|",
    ]

    ranking_key = "test_rounded_mae_all28_activecells"
    for name, values in sorted(metrics.items(), key=lambda item: item[1][ranking_key]):
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                [
                    fmt(values["test_rounded_mae_all28_activecells"]),
                    fmt(values["test_rounded_mae_all28_allcells"]),
                    fmt(values["test_rounded_rmse_all28_activecells"]),
                    fmt(values["test_rounded_rmse_all28_allcells"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Melhor Metodo Por Horizonte",
        "",
        "| Horizonte | Melhor metodo | MAE all cells |",
        "|---|---|---:|",
    ]
    for horizon in [1, 7, 14, 21, 28]:
        key = f"test_rounded_mae_h{horizon}_allcells"
        name, value = best_by_metric(metrics, key)
        lines.append(f"| h+{horizon} | {name} | {fmt(value)} |")

    lines += [
        "",
        "## Leitura",
        "",
        "- No agregado dos 28 dias, o melhor metodo atual por MAE e o STConvS2S-R Hurdle calibrado.",
        "- A calibracao de hiperparametros e threshold melhorou o Hurdle em relacao ao Poisson simples e ao Hurdle anterior.",
        "- A media movel de 7 dias ainda apresenta o melhor RMSE agregado, indicando menor penalizacao em erros grandes.",
        "- Persistencia segue competitiva em alguns horizontes curtos, mas perde no agregado.",
        "- O resultado principal do modelo calibrado usa `official-r-hurdle`, `pos_weight=10`, `occurrence_weight=1.0`, `count_weight=1.0`, checkpoint `best.pt` e threshold `0.87` selecionado em validacao.",
        "",
        "## Arquivos Fonte",
        "",
    ]
    for name, path in RUNS.items():
        lines.append(f"- {name}: `{path}`")

    (OUT_DIR / "comparison_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    metrics = load_metrics()
    write_csv(metrics)
    write_report(metrics)
    print(f"Comparison written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
