#!/usr/bin/env python3
"""Convert a complete sequence of monthly ERA5-Land files to weekly NPZ.

Monthly files are read one at a time, avoiding a multi-gigabyte concatenated
NetCDF. The command refuses incomplete month sequences and records source
metadata for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import num2date

try:
    from .process_era5_weekly import _merge_expver, _rh, _iso_week_monday
except ImportError:
    from process_era5_weekly import _merge_expver, _rh, _iso_week_monday


MONTH_RE = re.compile(r"(?P<year>\d{4})_(?P<month>\d{2})\.nc$", re.IGNORECASE)
CHANNELS = ["TEM_AVG", "TEM_MIN", "TEM_MAX", "RAIN", "RH_AVG", "RH_MIN", "RH_MAX"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_months(start: pd.Timestamp, end: pd.Timestamp):
    return [(value.year, value.month) for value in pd.period_range(start, end, freq="M")]


def load_month(path: Path, start: pd.Timestamp, end: pd.Timestamp):
    """Load one month and aggregate hourly ERA5 to seven daily channels."""
    dataset = nc.Dataset(path)
    try:
        lats = dataset.variables["latitude"][:].astype(np.float32)
        lons = dataset.variables["longitude"][:].astype(np.float32)
        time_name = "time" if "time" in dataset.variables else "valid_time"
        time_var = dataset.variables[time_name]
        timestamps = num2date(time_var[:], time_var.units)
        dates = np.asarray([datetime(t.year, t.month, t.day).date() for t in timestamps])
        selected = np.flatnonzero((dates >= start.date()) & (dates <= end.date()))
        if not len(selected):
            raise RuntimeError(f"No requested timestamps in {path}")
        t2m = _merge_expver(dataset.variables["t2m"][selected]).reshape(len(selected), len(lats), len(lons))
        d2m = _merge_expver(dataset.variables["d2m"][selected]).reshape(len(selected), len(lats), len(lons))
        rain = _merge_expver(dataset.variables["tp"][selected]).reshape(len(selected), len(lats), len(lons))
    finally:
        dataset.close()
    selected_dates = dates[selected]
    unique_days = sorted(set(selected_dates))
    daily = [np.full((len(unique_days), len(lats), len(lons)), np.nan, np.float32) for _ in range(7)]
    for index, day in enumerate(unique_days):
        hours = np.flatnonzero(selected_dates == day)
        humidity = _rh(t2m[hours], d2m[hours])
        daily[0][index] = np.nanmean(t2m[hours], axis=0) - 273.15
        daily[1][index] = np.nanmin(t2m[hours], axis=0) - 273.15
        daily[2][index] = np.nanmax(t2m[hours], axis=0) - 273.15
        daily[3][index] = np.nansum(np.maximum(rain[hours], 0), axis=0) * 1000.0
        daily[4][index] = np.nanmean(humidity, axis=0)
        daily[5][index] = np.nanmin(humidity, axis=0)
        daily[6][index] = np.nanmax(humidity, axis=0)
    return unique_days, daily, lats, lons


def aggregate_weekly(days, daily):
    groups = {}
    for index, day in enumerate(days):
        groups.setdefault(_iso_week_monday(day), []).append(index)
    weeks = sorted(groups)
    functions = [np.nanmean, np.nanmin, np.nanmax, np.nansum, np.nanmean, np.nanmin, np.nanmax]
    weekly = [np.full((len(weeks), *daily[0].shape[1:]), np.nan, np.float32) for _ in daily]
    for week_index, week in enumerate(weeks):
        indices = groups[week]
        for channel, function in enumerate(functions):
            weekly[channel][week_index] = function(daily[channel][indices], axis=0)
    return weeks, weekly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    directory = Path(args.input_dir)
    found = {}
    for path in directory.glob(f"{args.prefix}_????_??.nc"):
        match = MONTH_RE.search(path.name)
        if match:
            found[(int(match.group("year")), int(match.group("month")))] = path
    required = expected_months(start, end)
    missing = [f"{year:04d}-{month:02d}" for year, month in required if (year, month) not in found]
    if missing:
        raise RuntimeError(f"Monthly ERA5 sequence incomplete; missing {len(missing)}: {missing}")

    dates_all = []
    arrays = [[] for _ in range(7)]
    reference_lat = reference_lon = None
    sources = []
    for year, month in required:
        path = found[(year, month)]
        month_start = max(start, pd.Timestamp(year, month, 1))
        month_end = min(end, pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0))
        days, payload, lat, lon = load_month(path, month_start, month_end)
        if reference_lat is None:
            reference_lat, reference_lon = lat, lon
        elif not (np.array_equal(reference_lat, lat) and np.array_equal(reference_lon, lon)):
            raise RuntimeError(f"Grid mismatch in {path}")
        dates_all.extend(days)
        for target, values in zip(arrays, payload):
            target.append(values)
        sources.append({"file": str(path), "bytes": path.stat().st_size,
                        "sha256": None if args.skip_hashes else sha256(path)})

    if len(dates_all) != len(set(dates_all)):
        raise RuntimeError("Duplicate daily dates across monthly files")
    expected_days = pd.date_range(start, end, freq="D")
    if [value.isoformat() for value in dates_all] != [value.date().isoformat() for value in expected_days]:
        raise RuntimeError("Daily ERA5 sequence is not continuous over the requested interval")

    daily = [np.concatenate(parts, axis=0) for parts in arrays]
    week_dates, weekly = aggregate_weekly(dates_all, daily)
    values = np.stack(weekly, axis=1).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, era5_weekly=values, channels=np.asarray(CHANNELS),
                        channel_units=np.asarray(["degC", "degC", "degC", "mm", "%", "%", "%"]),
                        lat=reference_lat, lon=reference_lon,
                        week_dates=np.asarray([str(value) for value in week_dates]),
                        precipitation_conversion=np.asarray("ERA5 tp metres multiplied by 1000 to mm"))
    manifest = {"output": str(out), "shape": list(values.shape), "start": args.start,
                "end": args.end, "weeks": len(week_dates), "grid": [len(reference_lat), len(reference_lon)],
                "channels": CHANNELS, "units": ["degC", "degC", "degC", "mm", "%", "%", "%"],
                "monthly_files": sources, "test_2023_outcomes_used": False}
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("output", "shape", "start", "end", "weeks", "grid")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
