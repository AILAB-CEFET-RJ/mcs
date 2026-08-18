#!/usr/bin/env python3
"""
IDW (Inverse Distance Weighting) interpolation of station data onto the
ERA5 grid, with ERA5 as channel-level fallback where station coverage is sparse.

Output: {out}.npz  (same format as ERA5 weekly npz, with extra 'fill_ratio' key)

Usage:
  python src/data_handling/stations/interpolate_to_grid.py \
      --stations  data/processed/stations/RJ_stations_weekly.csv \
      --era5      data/processed/era5/RJ_weekly.npz \
      --out       data/processed/stations/RJ_grid_weekly.npz \
      --city      RJ \
      --alpha     2.0 \
      --cutoff-km 150
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Mapping from station CSV column → ERA5 channel index
CHANNEL_MAP = {
    "TEM_AVG": 0,
    "TEM_MIN": 1,
    "TEM_MAX": 2,
    "DEW_AVG": 3,
    "HUM_AVG": 4,
    "RAIN":    5,
}
ERA5_CHANNELS = ["TEM_AVG", "TEM_MIN", "TEM_MAX", "DEW_AVG", "RH_AVG", "PRECIP"]
# ERA5 channel names mapped to station channel names
ERA5_TO_STATION = {
    "TEM_AVG": "TEM_AVG",
    "TEM_MIN": "TEM_MIN",
    "TEM_MAX": "TEM_MAX",
    "DEW_AVG": "DEW_AVG",
    "RH_AVG":  "HUM_AVG",
    "PRECIP":  "RAIN",
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between arrays of points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def idw_weights(grid_lat, grid_lon, st_lats, st_lons, alpha=2.0, cutoff_km=150.0):
    """
    Compute IDW weight matrix W[h,w,s] for each grid cell and station.
    Stations beyond cutoff_km get zero weight.
    """
    if np.ndim(grid_lat) == 1 and np.ndim(grid_lon) == 1:
        lat_grid, lon_grid = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    elif np.shape(grid_lat) == np.shape(grid_lon) and np.ndim(grid_lat) == 2:
        lat_grid, lon_grid = np.asarray(grid_lat), np.asarray(grid_lon)
    else:
        raise ValueError("Grid coordinates must be matching 2-D arrays or two 1-D axes")
    H, W = lat_grid.shape
    S = len(st_lats)
    weights = np.zeros((H, W, S), dtype=np.float32)

    for s, (slat, slon) in enumerate(zip(st_lats, st_lons)):
        dist = haversine_km(lat_grid, lon_grid, slat, slon)
        dist[dist < 0.1] = 0.1  # avoid division by zero
        w = 1.0 / dist ** alpha
        w[dist > cutoff_km] = 0.0
        weights[:, :, s] = w.astype(np.float32)

    return weights


def interpolate_channel(weeks, station_vals, weights, *, method="idw", min_stations=1):
    """
    Interpolate one channel.

    Parameters
    ----------
    weeks        : (n_weeks,) array of week index
    station_vals : (n_weeks, S) array of station values (NaN where missing)
    weights      : (H, W, S) inverse-distance weights
    method       : ``idw`` or ``nearest``
    min_stations : minimum valid stations within the cutoff for each cell

    Returns
    -------
    grid : (n_weeks, H, W) float32
    used : (n_weeks,) bool — True if at least 1 station had valid data
    """
    n_weeks, S = station_vals.shape
    H, W, _ = weights.shape
    grid = np.full((n_weeks, H, W), np.nan, dtype=np.float32)
    used = np.zeros(n_weeks, dtype=bool)

    for wi in range(n_weeks):
        vals = station_vals[wi]         # (S,)
        valid = ~np.isnan(vals)
        if not valid.any():
            continue
        used[wi] = True
        v = vals[valid]                 # (n_valid,)
        w = weights[:, :, valid]        # (H, W, n_valid)
        station_count = (w > 0).sum(axis=2)
        no_coverage = station_count < int(min_stations)
        if method == "nearest":
            # The largest inverse-distance weight is the nearest currently
            # valid station. Cells failing the minimum are masked below.
            result = v[np.argmax(w, axis=2)].astype(np.float32)
        elif method == "idw":
            w_sum = w.sum(axis=2)
            safe_sum = np.where(w_sum > 0, w_sum, 1.0)
            numerator = (w * v[np.newaxis, np.newaxis, :]).sum(axis=2)
            result = numerator / safe_sum
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        result[no_coverage] = np.nan
        grid[wi] = result.astype(np.float32)

    return grid, used


def main():
    parser = argparse.ArgumentParser(description="IDW station → ERA5 grid with ERA5 fallback")
    parser.add_argument("--stations",  required=True)
    parser.add_argument("--era5",      required=True)
    parser.add_argument("--out",       required=True)
    parser.add_argument("--city",      default="RJ")
    parser.add_argument("--alpha",     type=float, default=2.0)
    parser.add_argument("--cutoff-km", type=float, default=150.0)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load ERA5 baseline
    log.info("Loading ERA5 baseline from %s", args.era5)
    era5 = np.load(args.era5, allow_pickle=True)
    era5_grid  = era5["era5_weekly"]         # (n_weeks, 6, H, W)
    era5_dates = era5["week_dates"]          # (n_weeks,) str YYYY-MM-DD
    era5_channels = list(era5["channels"])
    if "target_lat" in era5 and "target_lon" in era5:
        lats, lons = era5["target_lat"], era5["target_lon"]
        H, W = lats.shape
    else:
        lats, lons = era5["lat"], era5["lon"]
        H, W = len(lats), len(lons)
    n_weeks = era5_grid.shape[0]
    log.info("ERA5: %d weeks, H=%d W=%d, channels=%s", n_weeks, H, W, era5_channels)

    # Load station weekly
    log.info("Loading stations from %s", args.stations)
    st = pd.read_csv(args.stations)
    log.info("  %d rows, stations: %s", len(st), sorted(st["STATION"].unique()))

    # Build week → index map
    week_to_idx = {d: i for i, d in enumerate(era5_dates)}

    # Station metadata (unique per station)
    st_meta = st.drop_duplicates("STATION").set_index("STATION")[["LAT", "LNG"]].dropna()
    stations = st_meta.index.tolist()
    st_lats  = st_meta["LAT"].values.astype(float)
    st_lons  = st_meta["LNG"].values.astype(float)
    S = len(stations)
    log.info("Stations with coordinates: %d", S)

    # Compute IDW weights
    log.info("Computing IDW weights (alpha=%.1f, cutoff=%.0f km) …", args.alpha, args.cutoff_km)
    weights = idw_weights(lats, lons, st_lats, st_lons, args.alpha, args.cutoff_km)
    log.info("Weight matrix shape: %s", weights.shape)

    # Build per-channel station matrices
    result_grid = era5_grid.copy()  # start from ERA5
    fill_ratios = []

    for era5_ch, st_ch in ERA5_TO_STATION.items():
        ch_idx = era5_channels.index(era5_ch)
        if st_ch not in st.columns:
            log.warning("Station channel %s not found — using 100%% ERA5", st_ch)
            fill_ratios.append(1.0)
            continue

        # Build (n_weeks, S) station value matrix aligned to ERA5 weeks
        st_vals = np.full((n_weeks, S), np.nan, dtype=np.float32)
        for si, station_id in enumerate(stations):
            sub = st[st["STATION"] == station_id][["WEEK_DATE", st_ch]].dropna()
            for _, row in sub.iterrows():
                wi = week_to_idx.get(str(row["WEEK_DATE"]))
                if wi is not None:
                    st_vals[wi, si] = float(row[st_ch])

        log.info("Channel %s/%s: non-NaN station entries = %d/%d",
                 era5_ch, st_ch, (~np.isnan(st_vals)).sum(), n_weeks * S)

        idw_grid, used = interpolate_channel(np.arange(n_weeks), st_vals, weights)

        n_fallback = (~used).sum()
        fill_ratio = n_fallback / n_weeks
        fill_ratios.append(float(fill_ratio))
        log.info("  IDW coverage: %d/%d weeks  (ERA5 fallback: %.1f%%)",
                 used.sum(), n_weeks, 100 * fill_ratio)

        # Blend: use IDW where available, ERA5 elsewhere
        for wi in range(n_weeks):
            if used[wi]:
                cell = idw_grid[wi]
                missing_cells = np.isnan(cell)
                if missing_cells.any():
                    cell[missing_cells] = era5_grid[wi, ch_idx][missing_cells]
                result_grid[wi, ch_idx] = cell

    log.info("Saving → %s  shape=%s", args.out, result_grid.shape)
    output = dict(
        era5_weekly=result_grid.astype(np.float32),
        channels=np.array(era5_channels),
        lat=lats,
        lon=lons,
        week_dates=era5_dates,
        fill_ratio=np.array(fill_ratios, dtype=np.float32),
    )
    if np.ndim(lats) == 2:
        output["target_lat"] = lats
        output["target_lon"] = lons
    for audit_key in (
        "era5_source_pixel_ids",
        "interpolation_method",
        "resolution_warning",
    ):
        if audit_key in era5:
            output[audit_key] = era5[audit_key]
    np.savez_compressed(args.out, **output)
    log.info("Done.")


if __name__ == "__main__":
    main()
