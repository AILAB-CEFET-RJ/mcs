#!/usr/bin/env python3
"""Consolida exclusivamente os baselines e resultados metodologicamente corrigidos."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs" / "stage3_method_corrected_v2"
OUT = ROOT / "outputs" / "method_corrected_v2_comparison"
REQUIRED = ("model.pkl", "metrics.csv", "predictions.csv")

PATTERN = re.compile(
    r"^(RJ|RN)_(DAILY|WEEKLY)_(FULL|CASEONLY|CASESONLY)_"
    r"(BASELINE_METHOD_CORRECTED_V2|METHOD_CORRECTED_V2)_(\d+)_"
    r"(rf|xgb_poisson|xgb_zip)$"
)


def collect_metrics() -> pd.DataFrame:
    rows = []
    for folder in sorted(MODELS.iterdir()):
        if not folder.is_dir():
            continue
        match = PATTERN.fullmatch(folder.name)
        if not match:
            continue
        missing = [name for name in REQUIRED if not (folder / name).is_file()]
        if missing:
            raise RuntimeError(f"Execucao incompleta: {folder.name}: {missing}")
        local, resolution, features, stage, seed, model = match.groups()
        features = "CASEONLY" if features == "CASESONLY" else features
        metrics = pd.read_csv(folder / "metrics.csv")
        test = metrics.loc[metrics["Conjunto"] == "Teste"]
        if len(test) != 1:
            raise RuntimeError(f"Linha de teste invalida em {folder / 'metrics.csv'}")
        row = {
            "local": local,
            "resolution": resolution,
            "features": features,
            "stage": stage,
            "seed": int(seed),
            "model": model,
        }
        row.update(test.iloc[0].drop(labels=["Conjunto"]).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["stage", "local", "resolution", "features", "model"]
    value_cols = [c for c in metrics.columns if c not in keys + ["seed"]]
    grouped = metrics.groupby(keys, sort=True)[value_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).fillna(0).add_suffix("_std")
    count = grouped.size().rename("n_seeds")
    return pd.concat([count, mean, std], axis=1).reset_index()


def compare(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["local", "resolution", "features", "model"]
    base = summary[summary.stage == "BASELINE_METHOD_CORRECTED_V2"].drop(columns="stage")
    opt = summary[summary.stage == "METHOD_CORRECTED_V2"].drop(columns="stage")
    result = base.merge(opt, on=keys, suffixes=("_baseline", "_optimized"), validate="one_to_one")
    for metric in ("MSE", "RMSE", "MAE", "Poisson_Deviance"):
        before = result[f"{metric}_mean_baseline"]
        after = result[f"{metric}_mean_optimized"]
        result[f"{metric}_improvement_pct"] = 100 * (before - after) / before
    result["R2_change"] = result["R2_mean_optimized"] - result["R2_mean_baseline"]
    result["Spearman_change"] = result["Spearman_mean_optimized"] - result["Spearman_mean_baseline"]
    return result


def best_by_scenario(summary: pd.DataFrame) -> pd.DataFrame:
    optimized = summary[summary.stage == "METHOD_CORRECTED_V2"].copy()
    keys = ["local", "resolution", "features"]
    rows = []
    directions = {
        "RMSE_mean": "min",
        "MAE_mean": "min",
        "Poisson_Deviance_mean": "min",
        "R2_mean": "max",
        "Spearman_mean": "max",
    }
    for metric, direction in directions.items():
        grouped = optimized.groupby(keys, sort=True)[metric]
        indices = grouped.idxmin() if direction == "min" else grouped.idxmax()
        selected = optimized.loc[indices, keys + ["model", metric]].copy()
        selected.insert(3, "metric", metric.removesuffix("_mean"))
        selected = selected.rename(columns={"model": "best_model", metric: "best_mean"})
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).sort_values(keys + ["metric"])


def compare_feature_sets(summary: pd.DataFrame) -> pd.DataFrame:
    optimized = summary[summary.stage == "METHOD_CORRECTED_V2"].copy()
    keys = ["local", "resolution", "model"]
    values = [
        "RMSE_mean", "MAE_mean", "R2_mean",
        "Poisson_Deviance_mean", "Spearman_mean",
    ]
    full = optimized[optimized.features == "FULL"][keys + values]
    cases = optimized[optimized.features == "CASEONLY"][keys + values]
    result = full.merge(cases, on=keys, suffixes=("_full", "_caseonly"), validate="one_to_one")
    for metric in values:
        result[f"{metric}_full_minus_caseonly"] = (
            result[f"{metric}_full"] - result[f"{metric}_caseonly"]
        )
    return result


def collect_selected_trials() -> pd.DataFrame:
    rows = []
    for path in sorted(RUNS.glob("*/selected_trial.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "dataset": data["dataset"],
            "model": data["model"],
            "trial_number": data["trial_number"],
            "weighted_score": data["weighted_score"],
            "values": json.dumps(data["values"]),
            "params": json.dumps(data["params"], sort_keys=True),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics()
    expected = {"BASELINE_METHOD_CORRECTED_V2": 24, "METHOD_CORRECTED_V2": 240}
    counts = metrics.groupby("stage").size().to_dict()
    if counts != expected:
        raise RuntimeError(f"Quantidade inesperada de execucoes: {counts}; esperado: {expected}")
    summary = summarize(metrics)
    comparison = compare(summary)
    winners = best_by_scenario(summary)
    feature_comparison = compare_feature_sets(summary)
    selected = collect_selected_trials()
    if len(selected) != 24:
        raise RuntimeError(f"Quantidade inesperada de trials selecionados: {len(selected)}")
    metrics.to_csv(OUT / "metrics_by_seed.csv", index=False)
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    comparison.to_csv(OUT / "baseline_vs_optimized.csv", index=False)
    winners.to_csv(OUT / "best_model_by_scenario_and_metric.csv", index=False)
    feature_comparison.to_csv(OUT / "full_vs_caseonly.csv", index=False)
    selected.to_csv(OUT / "selected_trials.csv", index=False)
    print(f"Execucoes: {counts}")
    print(f"Cenarios comparados: {len(comparison)}")
    print(f"Trials selecionados: {len(selected)}")
    print(f"Saida: {OUT}")


if __name__ == "__main__":
    main()
