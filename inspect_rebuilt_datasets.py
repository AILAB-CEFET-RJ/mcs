"""Audita dimensões e suporte dos oito datasets reconstruídos."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "data" / "datasets"
NAMES = [
    "RJ_DAILY", "RJ_DAILY_CASESONLY",
    "RN_DAILY", "RN_DAILY_CASESONLY",
    "RJ_WEEKLY", "RJ_WEEKLY_CASESONLY",
    "RN_WEEKLY", "RN_WEEKLY_CASESONLY",
]
PAIRS = [
    ("RJ_DAILY", "RJ_DAILY_CASESONLY"),
    ("RN_DAILY", "RN_DAILY_CASESONLY"),
    ("RJ_WEEKLY", "RJ_WEEKLY_CASESONLY"),
    ("RN_WEEKLY", "RN_WEEKLY_CASESONLY"),
]


def load(name: str, file_name: str):
    with (ROOT / name / file_name).open("rb") as file:
        return pickle.load(file)


def main() -> None:
    for name in NAMES:
        arrays = load(name, "dataset.pickle")
        ids = load(name, "dataset_ids.pickle")
        print(f"\n{name}")
        for index, split in enumerate(("train", "val", "test")):
            X, y = arrays[2 * index], arrays[2 * index + 1]
            dates = pd.to_datetime(ids[split]["DATE"])
            units = np.unique(np.asarray(ids[split]["ID_UNIDADE"]).astype(str))
            print(
                f"  {split}: X={X.shape}; units={len(units)}; "
                f"dates={dates.min().date()}..{dates.max().date()}; "
                f"ysum={np.asarray(y).sum():.0f}; finite={np.isfinite(X).all()}"
            )

    for full, cases in PAIRS:
        full_ids = load(full, "dataset_ids.pickle")
        case_ids = load(cases, "dataset_ids.pickle")
        print(f"\nSUPPORT {full} / {cases}")
        for split in ("train", "val", "test"):
            full_dates = pd.to_datetime(full_ids[split]["DATE"]).to_numpy()
            case_dates = pd.to_datetime(case_ids[split]["DATE"]).to_numpy()
            full_units = np.asarray(full_ids[split]["ID_UNIDADE"]).astype(str)
            case_units = np.asarray(case_ids[split]["ID_UNIDADE"]).astype(str)
            same_keys = (
                np.array_equal(full_dates, case_dates)
                and np.array_equal(full_units, case_units)
            )
            same_targets = np.array_equal(
                np.asarray(full_ids[split]["Y_TRUE"]),
                np.asarray(case_ids[split]["Y_TRUE"]),
            )
            print(
                f"  {split}: keys={same_keys}; targets={same_targets}; "
                f"n={len(full_dates)}"
            )


if __name__ == "__main__":
    main()
