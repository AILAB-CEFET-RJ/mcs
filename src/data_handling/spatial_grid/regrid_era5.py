#!/usr/bin/env python3
"""Re-sample ERA5 coverage onto an independent epidemiological target grid.

This operation does not create new meteorological information.  The output
stores the contributing source-pixel ids for auditability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


def load_era5(path: str | Path):
    """Load native-grid ERA5 from NPZ or the project's long-form Parquet."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return np.load(path, allow_pickle=True)
    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("ERA5 input must be .npz or .parquet")
    frame = pd.read_parquet(path)
    required = {"DT_NOTIFIC", "LAT_ERA5", "LNG_ERA5"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"ERA5 Parquet missing columns: {sorted(missing)}")
    channels = [
        column for column in
        ["TEM_AVG", "TEM_MIN", "TEM_MAX", "RAIN", "RH_AVG", "RH_MIN", "RH_MAX"]
        if column in frame.columns
    ]
    dates = np.sort(pd.to_datetime(frame["DT_NOTIFIC"].dropna().unique()))
    lat = np.sort(frame["LAT_ERA5"].dropna().unique())
    lon = np.sort(frame["LNG_ERA5"].dropna().unique())
    full_index = pd.MultiIndex.from_product(
        [dates, lat, lon], names=["DT_NOTIFIC", "LAT_ERA5", "LNG_ERA5"]
    )
    indexed = frame.set_index(["DT_NOTIFIC", "LAT_ERA5", "LNG_ERA5"])
    if indexed.index.has_duplicates:
        raise ValueError("ERA5 Parquet has duplicate date/latitude/longitude rows")
    indexed = indexed.reindex(full_index)
    values = np.stack(
        [
            indexed[channel].to_numpy(dtype=np.float32).reshape(len(dates), len(lat), len(lon))
            for channel in channels
        ],
        axis=1,
    )
    return {
        "era5_weekly": values,
        "channels": np.asarray(channels),
        "week_dates": pd.to_datetime(dates).strftime("%Y-%m-%d").to_numpy(),
        "lat": lat,
        "lon": lon,
    }


def _ascending(axis: np.ndarray, data: np.ndarray, dimension: int):
    if axis[0] <= axis[-1]:
        return axis, data
    return axis[::-1], np.flip(data, axis=dimension)


def regrid(source: np.lib.npyio.NpzFile, cells: pd.DataFrame, method: str):
    data_key = "era5_daily" if "era5_daily" in source else "era5_weekly"
    values = source[data_key].astype(np.float32)
    src_lat, values = _ascending(np.asarray(source["lat"]), values, 2)
    src_lon, values = _ascending(np.asarray(source["lon"]), values, 3)
    rows = int(cells["row"].max()) + 1
    cols = int(cells["col"].max()) + 1
    ordered = cells.sort_values(["row", "col"])
    target_lat = ordered["latitude"].to_numpy().reshape(rows, cols)
    target_lon = ordered["longitude"].to_numpy().reshape(rows, cols)
    points = np.column_stack([target_lat.ravel(), target_lon.ravel()])
    out = np.empty((values.shape[0], values.shape[1], rows, cols), dtype=np.float32)
    nearest_fallback = np.zeros_like(out, dtype=bool)
    nearest_source_id = np.full(out.shape, -1, dtype=np.int16)
    src_lat_mesh, src_lon_mesh = np.meshgrid(src_lat, src_lon, indexing="ij")
    src_points = np.column_stack([src_lat_mesh.ravel(), src_lon_mesh.ravel()])
    for t in range(values.shape[0]):
        for channel in range(values.shape[1]):
            interpolator = RegularGridInterpolator(
                (src_lat, src_lon), values[t, channel], method=method,
                bounds_error=False, fill_value=np.nan,
            )
            interpolated = interpolator(points)
            missing = ~np.isfinite(interpolated)
            if missing.any():
                flat_source = values[t, channel].ravel()
                valid_source = np.isfinite(flat_source)
                if valid_source.any():
                    tree = cKDTree(src_points[valid_source])
                    _, nearest_local = tree.query(points[missing])
                    valid_ids = np.flatnonzero(valid_source)
                    source_ids = valid_ids[nearest_local]
                    interpolated[missing] = flat_source[source_ids]
                    nearest_fallback[t, channel].ravel()[missing] = True
                    nearest_source_id[t, channel].ravel()[missing] = source_ids.astype(np.int16)
            out[t, channel] = interpolated.reshape(rows, cols)

    lat_hi = np.searchsorted(src_lat, points[:, 0], side="left").clip(0, len(src_lat) - 1)
    lat_lo = (lat_hi - 1).clip(0)
    lon_hi = np.searchsorted(src_lon, points[:, 1], side="left").clip(0, len(src_lon) - 1)
    lon_lo = (lon_hi - 1).clip(0)
    contributors = np.column_stack(
        [
            lat_lo * len(src_lon) + lon_lo,
            lat_lo * len(src_lon) + lon_hi,
            lat_hi * len(src_lon) + lon_lo,
            lat_hi * len(src_lon) + lon_hi,
        ]
    ).reshape(rows, cols, 4)
    return out, target_lat, target_lon, contributors, nearest_fallback, nearest_source_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5", required=True)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["linear", "nearest"], default="linear")
    parser.add_argument("--target-id-column", default="", help="Persiste IDs associados aos pontos-alvo")
    args = parser.parse_args()
    source = load_era5(args.era5)
    data_key = "era5_daily" if "era5_daily" in source else "era5_weekly"
    date_key = "day_dates" if "day_dates" in source else "week_dates"
    cells = pd.read_csv(args.cells)
    values, lat, lon, contributors, nearest_fallback, nearest_source_id = regrid(source, cells, args.method)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        channels=source["channels"],
        target_lat=lat,
        target_lon=lon,
        era5_source_pixel_ids=contributors,
        era5_nearest_fallback_mask=nearest_fallback,
        era5_nearest_source_pixel_id=nearest_source_id,
        interpolation_method=np.asarray(args.method),
        channel_units=(
            source["channel_units"]
            if "channel_units" in source
            else np.asarray(["unknown"] * len(source["channels"]))
        ),
        precipitation_conversion=(
            source["precipitation_conversion"]
            if "precipitation_conversion" in source
            else np.asarray("not declared by source artifact")
        ),
        resolution_warning=np.asarray(
            "Re-sampled ERA5 coverage; no additional meteorological resolution"
        ),
    )
    payload[data_key] = values
    payload[date_key] = source[date_key]
    if args.target_id_column:
        if args.target_id_column not in cells:
            raise ValueError(f"Coluna de ID ausente: {args.target_id_column}")
        payload["target_ids"] = cells.sort_values(["row", "col"])[args.target_id_column].astype(str).to_numpy()
    np.savez_compressed(args.out, **payload)


if __name__ == "__main__":
    main()
