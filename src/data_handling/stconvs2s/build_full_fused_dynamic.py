#!/usr/bin/env python3
"""Build causal weekly FULL_FUSED dynamic channels for the statewide STConv grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ["TEM_AVG", "TEM_MIN", "TEM_MAX", "RAIN", "RH_AVG", "RH_MIN", "RH_MAX"]
ROLLING_WINDOWS = (1, 2, 3, 4)
BINARY = {"IDEAL_TEMP", "EXTREME_TEMP", "SIGNIFICANT_RAIN", "EXTREME_RAIN"}


def rolling(values: np.ndarray, window: int, operation: str) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for t in range(len(values)):
        start = max(0, t - window + 1)
        block = values[start : t + 1]
        out[t] = block.sum(axis=0, dtype=np.float64) if operation == "sum" else block.mean(axis=0, dtype=np.float64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", required=True, help="NPZ fusionado estadual (T,7,144,204)")
    parser.add_argument("--canonical", required=True, help="Tabela canônica semanal DATA x CNES")
    parser.add_argument("--epidemiology-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-end", default="2020-12-31")
    args = parser.parse_args()

    epi = Path(args.epidemiology_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    epi_dates = pd.DatetimeIndex(np.load(epi / "week_dates.npy").astype(str))
    cases = np.load(epi / "cases_weekly.npy", mmap_mode="r")
    with np.load(args.weather, allow_pickle=True) as source:
        data_key = "weather" if "weather" in source else "era5_weekly"
        date_key = "dates" if "dates" in source else "week_dates"
        weather_dates = pd.DatetimeIndex(source[date_key].astype(str))
        names = [str(v) for v in source["channels"]]
        units = [str(v) for v in source["channel_units"]]
        weather = np.asarray(source[data_key], dtype=np.float32)

    if names != BASE:
        raise ValueError(f"Canais meteorológicos inesperados: {names}")
    if units[names.index("RAIN")].lower() != "mm":
        raise ValueError(f"RAIN deve estar em mm, recebido {units[names.index('RAIN')]}")
    if weather.shape[0] != len(epi_dates) or weather.shape[-2:] != cases.shape[-2:]:
        raise ValueError(f"Forma incompatível: weather={weather.shape}, epidemiologia={cases.shape}")
    if not weather_dates.equals(epi_dates):
        raise ValueError("Datas meteorológicas e epidemiológicas não coincidem exatamente")
    if epi_dates.max().year > 2022:
        raise RuntimeError("Firewall violado: tensor de calibração contém 2023 ou posterior")
    if not np.isfinite(weather).all():
        raise ValueError("Campo meteorológico contém NaN/Inf após fallback ERA5")

    canonical_columns = ["DT_NOTIFIC", "row", "col", "CASES", *BASE]
    canonical = pd.read_parquet(args.canonical, columns=canonical_columns)
    canonical["DT_NOTIFIC"] = pd.to_datetime(canonical.DT_NOTIFIC)
    date_index = {date: index for index, date in enumerate(epi_dates)}
    reconstructed = np.zeros_like(cases, dtype=np.float32)
    grouped_cases = canonical.groupby(["DT_NOTIFIC", "row", "col"], as_index=False).CASES.sum()
    for item in grouped_cases.itertuples(index=False):
        reconstructed[date_index[pd.Timestamp(item.DT_NOTIFIC)], int(item.row), int(item.col)] = float(item.CASES)
    if not np.array_equal(reconstructed, np.asarray(cases)):
        raise RuntimeError("Transposição da tabela canônica não reproduz exatamente a grade de casos")
    observed = canonical.drop_duplicates(["DT_NOTIFIC", "row", "col"])
    for item in observed.itertuples(index=False):
        ti, row, col = date_index[pd.Timestamp(item.DT_NOTIFIC)], int(item.row), int(item.col)
        expected = weather[ti, :, row, col]
        received = np.asarray([getattr(item, name) for name in BASE], dtype=np.float32)
        if not np.allclose(expected, received, rtol=0, atol=1e-5):
            raise RuntimeError(f"Clima canônico diverge da grade em t={ti}, row={row}, col={col}")

    by_name = {name: weather[:, i] for i, name in enumerate(names)}
    temp_range = by_name["TEM_MAX"] - by_name["TEM_MIN"]
    feature_names = list(BASE) + [
        "IDEAL_TEMP", "EXTREME_TEMP", "SIGNIFICANT_RAIN", "EXTREME_RAIN",
        "TEMP_RANGE", "WEEK_OF_YEAR",
    ]
    values = [by_name[name] for name in BASE]
    values += [
        ((by_name["TEM_AVG"] >= 21) & (by_name["TEM_AVG"] <= 27)).astype(np.float32),
        ((by_name["TEM_AVG"] <= 14) | (by_name["TEM_AVG"] >= 38)).astype(np.float32),
        ((by_name["RAIN"] >= 10) & (by_name["RAIN"] < 150)).astype(np.float32),
        (by_name["RAIN"] >= 150).astype(np.float32),
        temp_range,
        np.broadcast_to(epi_dates.isocalendar().week.to_numpy(np.float32)[:, None, None], weather.shape[:1] + weather.shape[-2:]),
    ]
    for window in ROLLING_WINDOWS:
        derived = [
            (f"TEM_AVG_MM_{window}", rolling(by_name["TEM_AVG"], window, "mean")),
            (f"RAIN_ACC_{window}", rolling(by_name["RAIN"], window, "sum")),
            (f"RAIN_MM_{window}", rolling(by_name["RAIN"], window, "mean")),
            (f"RH_MM_{window}", rolling(by_name["RH_AVG"], window, "mean")),
            (f"TEMP_RANGE_MM_{window}", rolling(temp_range, window, "mean")),
        ]
        feature_names.extend(name for name, _ in derived)
        values.extend(value for _, value in derived)
    raw = np.stack(values, axis=1).astype(np.float32)
    train = epi_dates <= pd.Timestamp(args.train_end)
    mean = np.zeros(len(feature_names), dtype=np.float32)
    std = np.ones(len(feature_names), dtype=np.float32)
    scaled_path = out / "full_fused_dynamic_scaled.npy"
    scaled = np.lib.format.open_memmap(scaled_path, mode="w+", dtype=np.float32, shape=raw.shape)
    for index, name in enumerate(feature_names):
        if name not in BINARY:
            mean[index] = float(raw[train, index].mean(dtype=np.float64))
            std[index] = float(raw[train, index].std(dtype=np.float64)) or 1.0
        scaled[:, index] = (raw[:, index] - mean[index]) / std[index]
    scaled.flush()
    (out / "full_fused_dynamic_channels.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    np.savez(out / "full_fused_dynamic_scaler.npz", channels=np.asarray(feature_names), mean=mean, std=std)
    manifest = {
        "status": "READY", "shape": list(raw.shape), "channels": feature_names,
        "weather_source": str(Path(args.weather)), "canonical_table": str(Path(args.canonical)),
        "transposition_checks": {"cases_exact": True, "observed_cell_weather": True},
        "train_end": args.train_end,
        "dates": [str(epi_dates.min().date()), str(epi_dates.max().date())],
        "rain_units": "mm", "rain_thresholds_mm": [10, 150],
        "temperature_thresholds_c": {"ideal": [21, 27], "extreme": [14, 38]},
        "rolling_policy": "causal, current week and preceding weeks only",
        "year_2023_loaded": False,
    }
    (out / "full_fused_dynamic_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(scaled_path), "shape": list(raw.shape), "channels": len(feature_names)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
