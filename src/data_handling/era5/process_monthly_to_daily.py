#!/usr/bin/env python3
"""Build a continuous daily ERA5 artifact from the existing monthly files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import num2date

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_monthly_to_weekly import CHANNELS, MONTH_RE, expected_months, load_month  # noqa: E402
from process_era5_weekly import _merge_expver  # noqa: E402


def daily_precipitation(found, days, shape):
    """Sample D+1 00 UTC, which represents ERA5-Land accumulation during D."""
    output = np.full((len(days), *shape), np.nan, dtype=np.float32)
    requests = {}
    for index, day in enumerate(pd.DatetimeIndex(days)):
        stamp = day + pd.Timedelta(days=1)
        requests.setdefault((stamp.year, stamp.month), []).append((index, stamp))
    for key, items in requests.items():
        if key not in found:
            raise RuntimeError(f"Arquivo necessário para precipitação ausente: {key[0]:04d}-{key[1]:02d}")
        with nc.Dataset(found[key]) as dataset:
            time_name = "time" if "time" in dataset.variables else "valid_time"
            time = dataset.variables[time_name]
            timestamps = num2date(time[:], time.units)
            lookup = {(t.year, t.month, t.day, t.hour): i for i, t in enumerate(timestamps)}
            for out_index, stamp in items:
                source_index = lookup.get((stamp.year, stamp.month, stamp.day, 0))
                if source_index is None:
                    raise RuntimeError(f"00 UTC ausente em {found[key]} para {stamp.date()}")
                value = _merge_expver(dataset.variables["tp"][source_index]).reshape(shape)
                output[out_index] = np.maximum(value, 0) * 1000.0
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True); p.add_argument("--prefix", required=True)
    p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    start, end, directory = pd.Timestamp(args.start), pd.Timestamp(args.end), Path(args.input_dir)
    found = {}
    for path in directory.glob(f"{args.prefix}_????_??.nc"):
        match = MONTH_RE.search(path.name)
        if match: found[(int(match.group("year")), int(match.group("month")))] = path
    required = expected_months(start, end)
    precipitation_required = expected_months(start, end + pd.Timedelta(days=1))
    missing = [f"{y:04d}-{m:02d}" for y, m in required if (y, m) not in found]
    if missing: raise RuntimeError(f"Sequência mensal ERA5 incompleta: {missing}")
    dates, arrays, reference_lat, reference_lon = [], [[] for _ in CHANNELS], None, None
    for year, month in required:
        month_start = max(start, pd.Timestamp(year, month, 1))
        month_end = min(end, pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0))
        day, payload, lat, lon = load_month(found[(year, month)], month_start, month_end)
        if reference_lat is None: reference_lat, reference_lon = lat, lon
        elif not (np.array_equal(reference_lat, lat) and np.array_equal(reference_lon, lon)):
            raise RuntimeError("Grade ERA5 mudou entre arquivos mensais")
        dates.extend(day)
        for target, values in zip(arrays, payload): target.append(values)
    expected = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    if [str(v) for v in dates] != expected: raise RuntimeError("Série diária ERA5 descontínua")
    daily = [np.concatenate(parts) for parts in arrays]
    missing_precip = [f"{y:04d}-{m:02d}" for y, m in precipitation_required if (y, m) not in found]
    if missing_precip: raise RuntimeError(f"Sequência ERA5 incompleta para precipitação: {missing_precip}")
    daily[3] = daily_precipitation(found, expected, daily[0].shape[-2:])
    values = np.stack(daily, axis=1).astype(np.float32)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, era5_daily=values, channels=np.asarray(CHANNELS),
                        channel_units=np.asarray(["degC", "degC", "degC", "mm", "%", "%", "%"]),
                        lat=reference_lat, lon=reference_lon, day_dates=np.asarray(expected),
                        precipitation_conversion=np.asarray("ERA5-Land D+1 00 UTC assigned to D and multiplied by 1000 to mm"))
    manifest = {"status": "READY", "shape": list(values.shape), "start": args.start, "end": args.end,
                "frequency": "daily", "precipitation_policy": "D+1 00 UTC assigned to D",
                "year_2023_loaded_as_outcome": False}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__": main()
