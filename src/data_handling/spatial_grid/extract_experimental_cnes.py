#!/usr/bin/env python3
"""Extract the exact CNES universe used by an existing tabular dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True, help="dataset_ids.pickle")
    parser.add_argument("--cnes", required=True, help="CNES coordinate parquet")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ids = pd.read_pickle(args.ids)
    units = np.unique(
        np.concatenate([np.asarray(part["ID_UNIDADE"], dtype=str) for part in ids.values()])
    )
    units = pd.Series(units, name="CNES").str.replace(r"\.0$", "", regex=True).str.zfill(7)

    cnes = pd.read_parquet(args.cnes, columns=["CNES", "LAT", "LNG"]).copy()
    cnes["CNES"] = cnes["CNES"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
    cnes["LAT"] = pd.to_numeric(cnes["LAT"], errors="coerce")
    cnes["LNG"] = pd.to_numeric(cnes["LNG"], errors="coerce")
    cnes = cnes.dropna(subset=["LAT", "LNG"]).drop_duplicates("CNES", keep="first")
    selected = units.to_frame().merge(cnes, on="CNES", how="left", validate="one_to_one")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.dropna(subset=["LAT", "LNG"]).to_parquet(out, index=False)
    report = {
        "ids_source": str(Path(args.ids)),
        "coordinate_source": str(Path(args.cnes)),
        "units_in_dataset": int(len(units)),
        "units_with_coordinates": int(selected[["LAT", "LNG"]].notna().all(axis=1).sum()),
        "units_without_coordinates": selected.loc[
            selected[["LAT", "LNG"]].isna().any(axis=1), "CNES"
        ].tolist(),
        "output": str(out),
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
