#!/usr/bin/env python3
"""Fit per-channel epidemiological scalers using training instants only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from data.causal_epidemiological_dataset import (
        causal_epidemiological_frame,
        epidemiological_channel_names,
    )
except ImportError:
    from arboseer.src.data.causal_epidemiological_dataset import (
        causal_epidemiological_frame,
        epidemiological_channel_names,
    )


def fit_scaler(
    cases: np.ndarray,
    observation_mask: np.ndarray,
    time_indices: np.ndarray,
    lag_count: int,
    input_steps: int = 4,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute population mean/std once per unique time × observed cell."""
    indices = np.unique(np.asarray(time_indices, dtype=np.int64))
    if not len(indices):
        raise ValueError("nenhum instante de treino")
    if indices[-1] >= len(cases):
        raise ValueError("instante de treino fora da série")
    if not observation_mask.any():
        raise ValueError("máscara de observação vazia")
    n_channels = len(epidemiological_channel_names(lag_count, input_steps))
    sums = np.zeros(n_channels, dtype=np.float64)
    sums_sq = np.zeros(n_channels, dtype=np.float64)
    count = 0
    for time_index in indices:
        values = causal_epidemiological_frame(
            cases, int(time_index), lag_count, input_steps
        )
        selected = values[:, observation_mask].astype(np.float64, copy=False)
        sums += selected.sum(axis=1)
        sums_sq += np.square(selected).sum(axis=1)
        count += selected.shape[1]
    mean = sums / count
    variance = np.maximum(sums_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    # Canal constante não é informativo, mas deve permanecer numericamente seguro.
    std[std == 0] = 1.0
    return mean, std, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epidemiology-dir", required=True)
    parser.add_argument("--train-anchors", required=True)
    parser.add_argument("--max-lookback", type=int, default=12)
    parser.add_argument("--lag-count", type=int, choices=[4, 6, 8], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.epidemiology_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_2023_loaded") is not False:
        raise RuntimeError("firewall de 2023 não comprovado")
    cases = np.load(source / "cases_weekly.npy", mmap_mode="r")
    dates = np.load(source / "week_dates.npy", mmap_mode="r")
    if int(str(dates[-1])[:4]) > 2022:
        raise RuntimeError("firewall violado: scaler recebeu datas posteriores a 2022")
    anchors = np.load(args.train_anchors).astype(np.int64)
    if not len(anchors):
        raise ValueError("âncoras de treino vazias")
    # Instantes únicos efetivamente acessíveis pelo maior lookback candidato.
    first = int(anchors.min() - args.max_lookback + 1)
    last = int(anchors.max())
    time_indices = np.arange(first, last + 1, dtype=np.int64)
    if int(str(dates[last])[:4]) > 2020:
        raise RuntimeError("scaler de desenvolvimento deve terminar no treino de 2020")
    with np.load(source / "masks.npz") as masks:
        observation = masks["observation"].astype(bool)
    mean, std, count = fit_scaler(
        cases, observation, time_indices, args.lag_count, input_steps=4
    )
    names = epidemiological_channel_names(args.lag_count, input_steps=4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, channels=np.array(names), mean=mean, std=std,
        fitted_time_indices=time_indices,
    )
    report = {
        "status": "FIT_TRAIN_ONLY",
        "lag_count": args.lag_count,
        "channels": names,
        "observation_cells": int(observation.sum()),
        "unique_training_instants": int(len(time_indices)),
        "observations_per_channel": int(count),
        "first_date": str(dates[first]),
        "last_date": str(dates[last]),
        "year_2023_loaded": False,
        "target_scaled": False,
        "structural_channels_scaled": False,
        "output": str(out),
    }
    out.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
