#!/usr/bin/env python3
"""Calibrated per-network/per-channel weather fusion with ERA5 fallback.

Example rule::

    --rule WEBSIRENES:RAIN=idw,1,5,3

means IDW power 1, 5-km cutoff and at least three valid stations in each
cell/week. Channel priorities are independent, e.g.::

    --channel-priority RAIN=WEBSIRENES,ALERTARIO,INMET
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .fuse_weather_sources import CHANNEL_ALIASES, _canonical_column, _pick, _read_source
    from .interpolate_to_grid import idw_weights, interpolate_channel
except ImportError:
    from fuse_weather_sources import CHANNEL_ALIASES, _canonical_column, _pick, _read_source
    from interpolate_to_grid import idw_weights, interpolate_channel

LOG = logging.getLogger("fuse_weather_sources_calibrated")


@dataclass(frozen=True)
class Rule:
    method: str
    power: float
    cutoff_km: float
    min_stations: int
    min_completeness: float = 0.0


def parse_rule(value: str) -> tuple[tuple[str, str], Rule]:
    key, raw = value.split("=", 1)
    source, channel = [part.strip().upper() for part in key.split(":", 1)]
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in (4, 5):
        raise ValueError(f"Invalid rule {value}; expected method,power,cutoff_km,min_stations[,min_completeness]")
    rule = Rule(parts[0].lower(), float(parts[1]), float(parts[2]), int(parts[3]),
                float(parts[4]) if len(parts) == 5 else 0.0)
    if rule.method not in {"idw", "nearest"} or rule.cutoff_km <= 0 or rule.min_stations < 1:
        raise ValueError(f"Invalid rule values: {value}")
    if not 0 <= rule.min_completeness <= 1:
        raise ValueError("min_completeness must be between 0 and 1")
    return (source, channel), rule


def parse_priority(value: str) -> tuple[str, list[str]]:
    channel, raw = value.split("=", 1)
    names = [name.strip().upper() for name in raw.split(",") if name.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Repeated source in priority: {value}")
    return channel.strip().upper(), names


def validate_era5_units(era5, channels, allow_undeclared):
    if "channel_units" not in era5:
        if allow_undeclared:
            return ["undeclared"] * len(channels)
        raise ValueError("ERA5 artifact has no channel_units; refusing unit-unsafe fusion")
    units = [str(value) for value in era5["channel_units"]]
    if len(units) != len(channels):
        raise ValueError("ERA5 channel_units length differs from channels")
    if "RAIN" in channels and units[channels.index("RAIN")].lower() != "mm":
        raise ValueError(f"ERA5 RAIN must be mm, received {units[channels.index('RAIN')]}")
    return units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--rule", action="append", default=[], metavar="SOURCE:CHANNEL=METHOD,POWER,KM,MIN[,COMP]")
    parser.add_argument("--channel-priority", action="append", default=[], metavar="CHANNEL=SOURCE1,SOURCE2")
    parser.add_argument("--default-priority", default="INMET,ALERTARIO,WEBSIRENES")
    parser.add_argument("--allow-undeclared-era5-units", action="store_true")
    args = parser.parse_args()

    era5 = np.load(args.era5, allow_pickle=True)
    baseline, data_key = _pick(era5, ("weather", "era5_daily", "era5_weekly"))
    dates, dates_key = _pick(era5, ("dates", "day_dates", "week_dates"))
    channels = [str(value) for value in era5["channels"]]
    units = validate_era5_units(era5, channels, args.allow_undeclared_era5_units)
    result = np.asarray(baseline, dtype=np.float32).copy()
    if result.ndim != 4 or result.shape[1] != len(channels):
        raise ValueError(f"Expected ERA5 (T,C,H,W), received {result.shape}")
    lats = np.asarray(era5["target_lat"] if "target_lat" in era5 else era5["lat"], dtype=float)
    lons = np.asarray(era5["target_lon"] if "target_lon" in era5 else era5["lon"], dtype=float)
    date_strings = pd.to_datetime(dates).strftime("%Y-%m-%d")
    date_to_index = {value: index for index, value in enumerate(date_strings)}

    rules = dict(parse_rule(value) for value in args.rule)
    priorities = dict(parse_priority(value) for value in args.channel_priority)
    default_priority = [value.strip().upper() for value in args.default_priority.split(",") if value.strip()]
    source_paths = dict(_read_source(value) for value in args.source)
    unknown_priority = set(sum(priorities.values(), default_priority)).difference(source_paths)
    if unknown_priority:
        raise ValueError(f"Priority references unavailable sources: {sorted(unknown_priority)}")

    prepared = {}
    for name, path in source_paths.items():
        frame = pd.read_csv(path)
        missing = {"DATE", "STATION", "LAT", "LNG"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        frame["DATE"] = pd.to_datetime(frame.DATE, errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["DATE", "STATION", "LAT", "LNG"])
        metadata = frame.drop_duplicates("STATION").set_index("STATION")[["LAT", "LNG"]]
        prepared[name] = {"frame": frame, "metadata": metadata,
                          "stations": metadata.index.tolist(), "weights": {}}

    source_names = list(source_paths)
    source_codes = {name: index + 1 for index, name in enumerate(source_names)}
    provenance = np.zeros(result.shape, dtype=np.uint8)
    coverage = {name: {} for name in source_names}
    applied_rules = {}

    for channel_index, channel in enumerate(channels):
        priority = priorities.get(channel, default_priority)
        for source_name in priority:
            source = prepared[source_name]
            column = _canonical_column(source["frame"], channel)
            if column is None:
                continue
            rule = rules.get((source_name, channel), rules.get((source_name, "*")))
            if rule is None:
                raise ValueError(f"Missing interpolation rule for {source_name}:{channel}")
            applied_rules[f"{source_name}:{channel}"] = asdict(rule)
            completeness_candidates = [
                f"COMPLETENESS_{channel}",
                f"COMPLETENESS_{column}",
                "COMPLETENESS",
            ]
            completeness_column = next(
                (name for name in completeness_candidates if name in source["frame"]), None
            )
            if rule.min_completeness > 0 and completeness_column is None:
                raise ValueError(f"{source_name}:{channel} requires a channel-specific COMPLETENESS column")

            weight_key = (rule.power, rule.cutoff_km)
            if weight_key not in source["weights"]:
                meta = source["metadata"]
                source["weights"][weight_key] = idw_weights(
                    lats, lons, meta.LAT.to_numpy(float), meta.LNG.to_numpy(float),
                    alpha=max(rule.power, 1.0), cutoff_km=rule.cutoff_km,
                )
            weights = source["weights"][weight_key]
            values = np.full((len(dates), len(source["stations"])), np.nan, dtype=np.float32)
            station_index = {station: index for index, station in enumerate(source["stations"])}
            columns = ["DATE", "STATION", column]
            if completeness_column is not None:
                columns.append(completeness_column)
            subset = source["frame"][columns].copy()
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
            if completeness_column is not None:
                subset[completeness_column] = pd.to_numeric(
                    subset[completeness_column], errors="coerce"
                )
                subset.loc[
                    subset[completeness_column] < rule.min_completeness, column
                ] = np.nan
            subset = subset.dropna(subset=[column]).groupby(["DATE", "STATION"], as_index=False)[column].mean()
            for row in subset.itertuples(index=False):
                ti, si = date_to_index.get(row[0]), station_index.get(row[1])
                if ti is not None and si is not None:
                    values[ti, si] = float(row[2])
            interpolated, _ = interpolate_channel(
                np.arange(len(dates)), values, weights,
                method=rule.method, min_stations=rule.min_stations,
            )
            valid = np.isfinite(interpolated)
            available = int(valid.sum())
            unclaimed = provenance[:, channel_index] == 0
            use = valid & unclaimed
            result[:, channel_index][use] = interpolated[use]
            provenance[:, channel_index][use] = source_codes[source_name]
            coverage[source_name][channel] = {
                "eligible_values": available, "filled_values": int(use.sum()),
                "share_of_channel": float(use.sum() / use.size),
            }
            LOG.info("%s:%s filled %d values", source_name, channel, int(use.sum()))

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(weather=result, channels=np.asarray(channels), channel_units=np.asarray(units),
                  lat=lats, lon=lons, dates=np.asarray(date_strings), source_code=provenance,
                  source_names=np.asarray(["ERA5", *source_names]))
    for key in ("target_lat", "target_lon", "era5_source_pixel_ids", "era5_nearest_fallback_mask",
                "era5_nearest_source_pixel_id", "interpolation_method", "resolution_warning",
                "precipitation_conversion", "target_ids"):
        if key in era5:
            arrays[key] = era5[key]
    np.savez_compressed(output, **arrays)
    report = {
        "policy": "first valid source by channel-specific priority; ERA5 fallback",
        "era5": str(Path(args.era5)), "output": str(output), "shape": list(result.shape),
        "data_key_read": data_key, "dates_key_read": dates_key,
        "channel_units": dict(zip(channels, units)), "priorities": {
            channel: priorities.get(channel, default_priority) for channel in channels},
        "source_codes": {"ERA5": 0, **source_codes}, "rules": applied_rules,
        "coverage": coverage, "unit_guard_passed": True,
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
