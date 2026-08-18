#!/usr/bin/env python3
"""Wait for the resumable ERA5 download and build 5/10 km weekly products."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def run(command):
    print("RUN", subprocess.list2cmdline([str(value) for value in command]), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly-dir", required=True)
    parser.add_argument("--expected-files", type=int, default=132)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=48)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    arboseer = root / "arboseer"
    monthly = root / args.monthly_dir
    deadline = time.time() + args.timeout_hours * 3600
    while True:
        files = sorted(monthly.glob("RJ_STATE_????_??.nc"))
        print(f"WAIT {datetime.now(timezone.utc).isoformat()} files={len(files)}/{args.expected_files}", flush=True)
        if len(files) == args.expected_files:
            break
        if time.time() >= deadline:
            raise TimeoutError(f"ERA5 download incomplete after timeout: {len(files)}/{args.expected_files}")
        time.sleep(args.poll_seconds)

    native = arboseer / "data/processed/era5/RJ_STATE_native_weekly_2013_2023.npz"
    run([sys.executable, arboseer / "src/data_handling/era5/process_monthly_to_weekly.py",
         "--input-dir", monthly, "--prefix", "RJ_STATE", "--start", "2013-01-01",
         "--end", "2023-12-31", "--out", native])
    product = np.load(native, allow_pickle=True)
    rain_index = [str(value) for value in product["channels"]].index("RAIN")
    rain = product["era5_weekly"][:, rain_index]
    if not (0 < float(np.nanmedian(rain)) < 1000):
        raise RuntimeError(f"Implausible weekly ERA5 rain median in mm: {np.nanmedian(rain)}")

    outputs = []
    for label in ("2km", "5km", "10km"):
        cells = root / f"docs/calibracao_cap4/grade_1193/cells_{label}.csv"
        target = arboseer / f"data/processed/era5/RJ_STATE_{label}_weekly_2013_2023.npz"
        run([sys.executable, arboseer / "src/data_handling/spatial_grid/regrid_era5.py",
             "--era5", native, "--cells", cells, "--out", target, "--method", "linear"])
        outputs.append(str(target))
    manifest = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "monthly_files": len(files), "native": str(native), "regridded": outputs}
    destination = root / "docs/calibracao_cap4/era5_state_postprocess_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
