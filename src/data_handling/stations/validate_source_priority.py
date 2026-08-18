#!/usr/bin/env python3
"""Cross-network and ERA5 consistency audit for source-priority decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from .validate_interpolation import haversine, predict
except ImportError:
    from validate_interpolation import haversine, predict


def load_network(spec, end):
    name, path = spec.split("=", 1)
    frame = pd.read_csv(path)
    frame["DATE"] = pd.to_datetime(frame.DATE, errors="coerce")
    frame = frame.loc[frame.DATE <= end].dropna(subset=["DATE", "STATION", "LAT", "LNG"])
    return name, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--era5", required=True)
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    end = pd.Timestamp(args.end_date)
    networks = dict(load_network(spec, end) for spec in args.source)
    selected = pd.read_csv(args.selected)
    records = []

    # Cross-network rainfall: observations in one network are predicted using
    # the configuration calibrated internally in each other network.
    for target_name, target in networks.items():
        if "RAIN" not in target:
            continue
        target = target.dropna(subset=["RAIN"])
        for source_name, source in networks.items():
            if source_name == target_name or "RAIN" not in source:
                continue
            config = selected[(selected.network == source_name) & (selected.variable == "RAIN")]
            if config.empty:
                continue
            config = config.iloc[0]
            by_date = {date: group for date, group in source.dropna(subset=["RAIN"]).groupby("DATE")}
            for row in target.itertuples(index=False):
                candidates = by_date.get(row.DATE)
                if candidates is None:
                    continue
                distance = haversine(row.LAT, row.LNG, candidates.LAT.to_numpy(float),
                                     candidates.LNG.to_numpy(float))
                estimate = predict(candidates.RAIN.to_numpy(float), distance,
                                   method=str(config.method), power=float(config.power),
                                   radius=float(config.radius), minimum=int(config.minimum))
                if np.isfinite(estimate):
                    records.append({"target": target_name, "source": source_name,
                                    "variable": "RAIN", "observed": float(row.RAIN),
                                    "predicted": estimate, "date": row.DATE})

    # ERA5 nearest-cell consistency at station coordinates. The available
    # artifact is municipal; targets outside its extent are naturally absent.
    era5 = np.load(args.era5, allow_pickle=True)
    values = era5["era5_weekly"]
    channels = [str(value) for value in era5["channels"]]
    dates = pd.to_datetime(era5["week_dates"])
    date_index = {date: i for i, date in enumerate(dates) if date <= end}
    lat, lon = era5["target_lat"], era5["target_lon"]
    points = np.column_stack([lat.ravel(), lon.ravel()])
    tree = cKDTree(points)
    bounds = (lon.min(), lat.min(), lon.max(), lat.max())
    aliases = {"HUM_AVG": "RH_AVG"}
    for target_name, target in networks.items():
        for variable in [v for v in ("TEM_AVG", "TEM_MIN", "TEM_MAX", "RAIN", "HUM_AVG", "RH_AVG") if v in target]:
            era_channel = aliases.get(variable, variable)
            if era_channel not in channels:
                continue
            ci = channels.index(era_channel)
            for row in target.dropna(subset=[variable]).itertuples(index=False):
                ti = date_index.get(row.DATE)
                if ti is None or not (bounds[0] <= row.LNG <= bounds[2] and bounds[1] <= row.LAT <= bounds[3]):
                    continue
                _, gi = tree.query([row.LAT, row.LNG])
                estimate = float(values[ti, ci].ravel()[gi])
                records.append({"target": target_name, "source": "ERA5",
                                "variable": era_channel, "observed": float(getattr(row, variable)),
                                "predicted": estimate, "date": row.DATE})

    raw = pd.DataFrame(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "source_comparisons_raw.csv", index=False)
    summary = []
    for key, group in raw.groupby(["target", "source", "variable"]):
        error = group.predicted - group.observed
        summary.append({"target": key[0], "source": key[1], "variable": key[2],
                        "n": len(group), "mae": error.abs().mean(),
                        "rmse": np.sqrt(np.mean(error**2)), "bias": error.mean(),
                        "start": group.date.min(), "end": group.date.max()})
    summary = pd.DataFrame(summary).sort_values(["variable", "target", "mae"])
    summary.to_csv(out / "source_comparisons_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
