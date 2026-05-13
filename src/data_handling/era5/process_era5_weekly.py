#!/usr/bin/env python3
"""
Convert ERA5-Land hourly NetCDF → weekly (ISO week Monday) NPZ grid.

Outputs RJ_weekly.npz with arrays:
  era5_weekly : float32 (n_weeks, 6, H, W)
  channels    : list[str]  ['TEM_AVG','TEM_MIN','TEM_MAX','DEW_AVG','RH_AVG','PRECIP']
  lat         : float32 (H,)
  lon         : float32 (W,)
  week_dates  : str (n_weeks,)  ISO week start dates YYYY-MM-DD (Mondays)

Usage:
  python src/data_handling/era5/process_era5_weekly.py \
      --file  data/raw/era5/RJ_1997_2024.nc \
      --start 2013-12-30 \
      --end   2023-12-31 \
      --out   data/processed/era5/RJ_weekly.npz
"""

import argparse
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from netCDF4 import num2date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHANNELS = ["TEM_AVG", "TEM_MIN", "TEM_MAX", "DEW_AVG", "RH_AVG", "PRECIP"]


def _rh(t_k, td_k):
    """Relative humidity (%) from temperature and dew point in Kelvin."""
    t  = t_k  - 273.15
    td = td_k - 273.15
    return 100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))


def _merge_expver(arr):
    """
    ERA5 files downloaded with 2 experiment versions have shape (..., 2).
    Merge by taking expver=0 where valid, filling gaps from expver=1.
    """
    if arr.ndim == 0 or arr.shape[-1] != 2:
        return arr
    a0 = arr[..., 0].astype(np.float32)
    a1 = arr[..., 1].astype(np.float32)
    # values of 9.969e+36 (netCDF fill) → NaN
    fill = 9.96921e36
    a0[a0 > fill * 0.9] = np.nan
    a1[a1 > fill * 0.9] = np.nan
    result = np.where(np.isnan(a0), a1, a0)
    return result


def _iso_week_monday(dt):
    """Return the ISO week Monday (as date) for a datetime."""
    d = dt.date() if hasattr(dt, "date") else dt
    return d - timedelta(days=d.weekday())


def load_era5(path, start_date, end_date):
    """Load t2m, d2m, tp from NetCDF between start/end dates (inclusive).

    Returns
    -------
    dates  : list of Python date objects (length = n_days)
    t2m_d  : (n_days, H, W) float32  daily-mean temperature K
    t2m_min: (n_days, H, W) float32  daily-min temperature K
    t2m_max: (n_days, H, W) float32  daily-max temperature K
    d2m_d  : (n_days, H, W) float32  daily-mean dew point K
    rh_d   : (n_days, H, W) float32  daily-mean relative humidity %
    tp_d   : (n_days, H, W) float32  daily-sum precipitation mm
    lats   : (H,) float32
    lons   : (W,) float32
    """
    log.info("Opening %s", path)
    ds = nc.Dataset(path)

    lats = ds.variables["latitude"][:].astype(np.float32)
    lons = ds.variables["longitude"][:].astype(np.float32)
    H, W = len(lats), len(lons)

    time_var = ds.variables["time"]
    all_times = num2date(time_var[:], time_var.units)

    # Filter to requested window
    sd = datetime.strptime(start_date, "%Y-%m-%d").date()
    ed = datetime.strptime(end_date,   "%Y-%m-%d").date()

    def _to_date(t):
        return datetime(t.year, t.month, t.day).date()

    sel = np.array([sd <= _to_date(t) <= ed for t in all_times])
    idx = np.where(sel)[0]
    if len(idx) == 0:
        raise ValueError(f"No ERA5 timesteps found between {start_date} and {end_date}")
    log.info("Selected %d hourly timesteps (%s -> %s)",
             len(idx), all_times[idx[0]], all_times[idx[-1]])

    log.info("Loading t2m …")
    t2m_raw = ds.variables["t2m"][idx]
    t2m = _merge_expver(t2m_raw).reshape(len(idx), H, W)

    log.info("Loading d2m …")
    d2m_raw = ds.variables["d2m"][idx]
    d2m = _merge_expver(d2m_raw).reshape(len(idx), H, W)

    log.info("Loading tp  …")
    tp_raw = ds.variables["tp"][idx]
    tp = _merge_expver(tp_raw).reshape(len(idx), H, W)

    ds.close()

    # Build date index for grouping
    hour_dates = [_to_date(all_times[i]) for i in idx]
    unique_days = sorted(set(hour_dates))

    day_to_hours = {d: [] for d in unique_days}
    for k, d in enumerate(hour_dates):
        day_to_hours[d].append(k)

    n_days = len(unique_days)
    t2m_d   = np.full((n_days, H, W), np.nan, dtype=np.float32)
    t2m_min = np.full((n_days, H, W), np.nan, dtype=np.float32)
    t2m_max = np.full((n_days, H, W), np.nan, dtype=np.float32)
    d2m_d   = np.full((n_days, H, W), np.nan, dtype=np.float32)
    rh_d    = np.full((n_days, H, W), np.nan, dtype=np.float32)
    tp_d    = np.full((n_days, H, W), np.nan, dtype=np.float32)

    log.info("Aggregating %d days …", n_days)
    for di, day in enumerate(unique_days):
        hrs = day_to_hours[day]
        t2m_d[di]   = np.nanmean(t2m[hrs], axis=0)
        t2m_min[di] = np.nanmin(t2m[hrs],  axis=0)
        t2m_max[di] = np.nanmax(t2m[hrs],  axis=0)
        d2m_d[di]   = np.nanmean(d2m[hrs], axis=0)
        rh_d[di]    = np.nanmean(_rh(t2m[hrs], d2m[hrs]), axis=0)
        tp_d[di]    = np.nansum(np.maximum(tp[hrs], 0), axis=0) * 1000.0  # m → mm

    return unique_days, t2m_d, t2m_min, t2m_max, d2m_d, rh_d, tp_d, lats, lons


def daily_to_weekly(days, *daily_arrays):
    """Group daily arrays by ISO week (Monday start).

    Returns
    -------
    week_mondays : list of date
    weekly_arrays: tuple of (n_weeks, H, W) arrays per input
        TEM_AVG, DEW_AVG, RH_AVG: weekly mean
        TEM_MIN: weekly min
        TEM_MAX: weekly max
        tp_d: weekly sum
    """
    week_map = {}
    for di, d in enumerate(days):
        mon = _iso_week_monday(d)
        week_map.setdefault(mon, []).append(di)

    week_mondays = sorted(week_map)
    n_weeks = len(week_mondays)
    H = daily_arrays[0].shape[1]
    W = daily_arrays[0].shape[2]

    out = [np.full((n_weeks, H, W), np.nan, dtype=np.float32) for _ in daily_arrays]

    for wi, mon in enumerate(week_mondays):
        hrs = week_map[mon]
        # t2m_d → mean, t2m_min → min, t2m_max → max, d2m_d → mean, rh_d → mean, tp_d → sum
        agg_fns = [np.nanmean, np.nanmin, np.nanmax, np.nanmean, np.nanmean, np.nansum]
        for ai, (arr, fn) in enumerate(zip(daily_arrays, agg_fns)):
            out[ai][wi] = fn(arr[hrs], axis=0)

    return week_mondays, out


def main():
    parser = argparse.ArgumentParser(description="ERA5 hourly → weekly grid NPZ")
    parser.add_argument("--file",  required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="YYYY-MM-DD")
    parser.add_argument("--out",   required=True, help="Output .npz path")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    days, t2m_d, t2m_min, t2m_max, d2m_d, rh_d, tp_d, lats, lons = load_era5(
        args.file, args.start, args.end
    )

    week_mondays, (wt_avg, wt_min, wt_max, wd_avg, wrh, wtp) = daily_to_weekly(
        days, t2m_d, t2m_min, t2m_max, d2m_d, rh_d, tp_d
    )

    # Convert K → °C for temperature channels
    era5_weekly = np.stack([
        wt_avg - 273.15,  # TEM_AVG
        wt_min - 273.15,  # TEM_MIN
        wt_max - 273.15,  # TEM_MAX
        wd_avg - 273.15,  # DEW_AVG
        wrh,              # RH_AVG
        wtp,              # PRECIP
    ], axis=1).astype(np.float32)  # (n_weeks, 6, H, W)

    week_dates = np.array([str(m) for m in week_mondays])

    log.info("Saving → %s  shape=%s", args.out, era5_weekly.shape)
    np.savez_compressed(
        args.out,
        era5_weekly=era5_weekly,
        channels=np.array(CHANNELS),
        lat=lats,
        lon=lons,
        week_dates=week_dates,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
