#!/usr/bin/env python3
"""
Inspect an ERA5-Land NetCDF file: grid dimensions, variable names,
time coverage, and optional bounding-box check.

Usage:
  python src/data_handling/era5/inspect_era5.py \
      --file data/raw/era5/RJ_1997_2024.nc \
      --check-bbox -23.5 -21.7 -44.9 -40.9 \
      --time-slice 2013-12-30 2023-12-31
"""

import argparse
import sys

import netCDF4 as nc
import numpy as np
from netCDF4 import num2date


def main():
    parser = argparse.ArgumentParser(description="Inspect ERA5-Land NetCDF file")
    parser.add_argument("--file", required=True, help="Path to .nc file")
    parser.add_argument("--check-bbox", nargs=4, type=float,
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        help="Assert grid covers this bounding box")
    parser.add_argument("--time-slice", nargs=2, metavar=("START", "END"),
                        help="Assert time coverage includes these dates (YYYY-MM-DD)")
    args = parser.parse_args()

    ds = nc.Dataset(args.file)
    print(f"File : {args.file}")
    print(f"Format: {ds.data_model}")

    lats = ds.variables["latitude"][:]
    lons = ds.variables["longitude"][:]
    H, W = len(lats), len(lons)
    print(f"\nGrid : H={H}, W={W}")
    print(f"  Lat range : {float(lats.min()):.4f} -> {float(lats.max()):.4f}")
    print(f"  Lon range : {float(lons.min()):.4f} -> {float(lons.max()):.4f}")
    print(f"  Lat step  : {float(np.diff(np.sort(lats)).mean()):.4f}")
    print(f"  Lon step  : {float(np.diff(np.sort(lons)).mean()):.4f}")

    time_var = ds.variables["time"]
    n_times = len(time_var)
    t0 = num2date(time_var[0],  time_var.units)
    t1 = num2date(time_var[-1], time_var.units)
    print(f"\nTime  : {n_times} steps")
    print(f"  First : {t0}")
    print(f"  Last  : {t1}")
    print(f"  Units : {time_var.units}")

    print("\nVariables:")
    for vname, var in ds.variables.items():
        if vname in ("latitude", "longitude", "time", "expver"):
            continue
        units = getattr(var, "units", "?")
        long  = getattr(var, "long_name", "")
        print(f"  {vname:8s} shape={var.shape}  units={units}  ({long})")

    if "expver" in ds.variables:
        expver_vals = ds.variables["expver"][:]
        print(f"\nexpver values: {expver_vals.tolist()}")
        print("  (multiple expver — will merge with first-non-NaN strategy)")

    ok = True
    if args.check_bbox:
        lat_min, lat_max, lon_min, lon_max = args.check_bbox
        if lats.min() > lat_min or lats.max() < lat_max:
            print(f"\n[WARN] Lat bbox mismatch: grid [{lats.min():.2f},{lats.max():.2f}]"
                  f" vs requested [{lat_min},{lat_max}]")
            ok = False
        if lons.min() > lon_min or lons.max() < lon_max:
            print(f"[WARN] Lon bbox mismatch: grid [{lons.min():.2f},{lons.max():.2f}]"
                  f" vs requested [{lon_min},{lon_max}]")
            ok = False
        if ok:
            print(f"\n[OK] Grid covers bbox [{lat_min},{lat_max}] x [{lon_min},{lon_max}]")

    if args.time_slice:
        from datetime import datetime
        req_start = datetime.strptime(args.time_slice[0], "%Y-%m-%d")
        req_end   = datetime.strptime(args.time_slice[1], "%Y-%m-%d")
        grid_start = datetime(t0.year, t0.month, t0.day)
        grid_end   = datetime(t1.year, t1.month, t1.day)
        if grid_start > req_start or grid_end < req_end:
            print(f"[WARN] Time coverage [{grid_start.date()} -> {grid_end.date()}]"
                  f" does not fully cover [{req_start.date()} -> {req_end.date()}]")
        else:
            print(f"[OK] Time coverage includes {req_start.date()} -> {req_end.date()}")

    ds.close()
    print(f"\nH={H}, W={W}  (use these for tensor shape verification in Step 5)")


if __name__ == "__main__":
    main()
