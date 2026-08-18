#!/usr/bin/env python3
"""Leave-one-station-out validation for weekly meteorological networks.

The routine uses no dengue outcomes and excludes dates after ``--end-date``.
Each station observation is predicted only from other stations in the same
network and week. Results are emitted per network, variable and candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES = ("TEM_AVG", "TEM_MIN", "TEM_MAX", "RH_AVG", "HUM_AVG", "RAIN")


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def predict(values, distances, method, power, radius, minimum):
    eligible = np.isfinite(values) & np.isfinite(distances) & (distances <= radius)
    if int(eligible.sum()) < minimum:
        return np.nan
    vals, dist = values[eligible], np.maximum(distances[eligible], 0.1)
    if method == "nearest":
        return float(vals[np.argmin(dist)])
    weights = 1.0 / np.power(dist, power)
    return float(np.sum(weights * vals) / np.sum(weights))


def evaluate(frame, variable, candidates):
    observations = []
    for _, weekly in frame.dropna(subset=[variable]).groupby("DATE"):
        if len(weekly) < 2:
            continue
        lat = weekly["LAT"].to_numpy(float)
        lon = weekly["LNG"].to_numpy(float)
        value = weekly[variable].to_numpy(float)
        station = weekly["STATION"].astype(str).to_numpy()
        for index in range(len(weekly)):
            keep = np.arange(len(weekly)) != index
            distances = haversine(lat[index], lon[index], lat[keep], lon[keep])
            for candidate in candidates:
                estimate = predict(value[keep], distances, **candidate)
                if np.isfinite(estimate):
                    observations.append(
                        {**candidate, "date": weekly["DATE"].iloc[0], "station": station[index],
                         "observed": value[index], "predicted": estimate}
                    )
    return pd.DataFrame(observations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    candidates = []
    for radius in (5.0, 10.0, 20.0):
        for minimum in (1, 2, 3):
            candidates.append(dict(method="nearest", power=0.0, radius=radius, minimum=minimum))
            candidates.append(dict(method="idw", power=1.0, radius=radius, minimum=minimum))
            candidates.append(dict(method="idw", power=2.0, radius=radius, minimum=minimum))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, availability = [], []
    for spec in args.source:
        name, raw = spec.split("=", 1)
        frame = pd.read_csv(raw)
        frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce")
        frame = frame.loc[frame["DATE"] <= pd.Timestamp(args.end_date)].copy()
        frame["LAT"] = pd.to_numeric(frame["LAT"], errors="coerce")
        frame["LNG"] = pd.to_numeric(frame["LNG"], errors="coerce")
        frame = frame.dropna(subset=["DATE", "STATION", "LAT", "LNG"])
        present = [variable for variable in VARIABLES if variable in frame.columns]
        availability.append({"network": name, "rows": len(frame), "stations": frame.STATION.nunique(),
                             "start": str(frame.DATE.min().date()) if len(frame) else None,
                             "end": str(frame.DATE.max().date()) if len(frame) else None,
                             "variables": present})
        for variable in present:
            raw_predictions = evaluate(frame, variable, candidates)
            if raw_predictions.empty:
                continue
            raw_predictions.to_csv(out / f"predictions_{name}_{variable}.csv", index=False)
            total_targets = int(frame[variable].notna().sum())
            keys = ["method", "power", "radius", "minimum"]
            for key, group in raw_predictions.groupby(keys, dropna=False):
                error = group.predicted - group.observed
                summaries.append({"network": name, "variable": variable,
                    **dict(zip(keys, key)), "n_predictions": len(group),
                    "n_targets": total_targets, "coverage_pct": 100 * len(group) / total_targets,
                    "mae": float(error.abs().mean()), "rmse": float(np.sqrt(np.mean(error**2))),
                    "bias": float(error.mean()), "stations_evaluated": int(group.station.nunique())})

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "loocv_summary.csv", index=False)
    pd.DataFrame(availability).to_json(out / "availability.json", orient="records", indent=2)
    if len(summary):
        selected = []
        for _, group in summary.groupby(["network", "variable"], sort=True):
            eligible = group.loc[group.coverage_pct >= 80]
            pool = eligible if len(eligible) else group
            selected.append(pool.sort_values(
                ["mae", "rmse", "coverage_pct"], ascending=[True, True, False]
            ).head(1))
        best = pd.concat(selected, ignore_index=True)
        best.to_csv(out / "selected_candidates.csv", index=False)
        print(best.to_string(index=False))
    (out / "manifest.json").write_text(json.dumps({
        "end_date": args.end_date, "sources": args.source,
        "selection": "minimum MAE among candidates with >=80% coverage; fallback to all candidates",
        "dengue_outcomes_used": False, "test_2023_used": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
