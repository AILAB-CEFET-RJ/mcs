import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "data_handling"))

from aggregate import aggregate_by_date
from data_handling.build_dataset import (
    aggregate_era5_grid,
    build_partition_with_history,
    extract_era5_data,
    filter_to_reference_support,
)
from data_handling.features.feature_engineering import create_new_features
from evaluation.eval_utils import get_optimization_metrics


class PipelineRegressionTests(unittest.TestCase):
    def test_vectorized_era5_matches_legacy_extraction(self):
        times = pd.date_range("2020-01-06", periods=24 * 7, freq="h")
        shape = (len(times), 1, 1)
        t2m = np.linspace(295.0, 305.0, np.prod(shape)).reshape(shape)
        ds = xr.Dataset(
            {
                "t2m": (("time", "latitude", "longitude"), t2m),
                "d2m": (("time", "latitude", "longitude"), t2m - 3.0),
                "tp": (("time", "latitude", "longitude"), np.full(shape, 0.001)),
            },
            coords={"time": times, "latitude": [-5.75], "longitude": [-35.25]},
        )
        config = SimpleNamespace(
            weekly=True,
            features={"enable": {"raw_features": {
                "tem_avg": True, "tem_min": True, "tem_max": True,
                "rain": True, "rh_avg": True, "rh_min": True, "rh_max": True,
            }}},
        )
        legacy = extract_era5_data(ds, -5.75, -35.25, times[0], config)
        vectorized = aggregate_era5_grid(ds, times[0], times[0], weekly=True).iloc[0]
        for column, expected in legacy.items():
            self.assertAlmostEqual(float(vectorized[column]), float(expected), places=10)

    def test_temporal_features_do_not_cross_unit_boundaries(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "ID_UNIDADE": ["B", "A", "B", "A", "B", "A"],
            "DT_NOTIFIC": [dates[1], dates[0], dates[0], dates[2], dates[2], dates[1]],
            "CASES": [200, 1, 100, 3, 300, 2],
        })
        config = SimpleNamespace(
            min_date="2020-01-01",
            max_date="2020-01-03",
            features={
                "enable": {
                    "cases_windows": True,
                    "cases_accumulators": False,
                    "cases_lags": True,
                    "windows": False,
                },
                "windows": [2],
                "lags": 1,
            },
        )

        with tempfile.TemporaryDirectory() as output_dir:
            X, y, sidecar = create_new_features(df, "train", config, output_dir)
            names = pd.read_csv(Path(output_dir) / "feature_dictionary.csv")["Feature"].tolist()

        mm_idx = names.index("CASES_MM_2")
        lag_idx = names.index("CASES_LAG_1")
        rows = {
            unit: (X[index, mm_idx], X[index, lag_idx], y[index])
            for index, unit in enumerate(sidecar["ID_UNIDADE"])
        }
        self.assertEqual(rows["A"], (1.5, 1.0, 3.0))
        self.assertEqual(rows["B"], (150.0, 100.0, 300.0))

    def test_optimization_metrics_have_minimization_direction(self):
        metrics = get_optimization_metrics(
            np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])
        )
        self.assertEqual(list(metrics), [
            "MSE", "RMSE", "MAE", "Negative_R2",
            "MAPE_ignoring_zeros", "SMAPE", "Negative_Spearman",
            "Poisson_Deviance",
        ])
        self.assertEqual(metrics["Negative_R2"], -1.0)
        self.assertEqual(metrics["Negative_Spearman"], -1.0)

    def test_validation_can_use_strictly_past_context(self):
        dates = pd.date_range("2020-01-06", periods=8, freq="W-MON")
        df = pd.DataFrame({
            "ID_UNIDADE": ["A"] * len(dates),
            "DT_NOTIFIC": dates,
            "CASES": np.arange(1, len(dates) + 1, dtype=float),
        })
        split = pd.Timestamp("2020-02-03")
        contextual = build_partition_with_history(df, split, None, context_rows=4)
        config = SimpleNamespace(
            min_date="2020-01-06",
            max_date="2020-02-24",
            features={
                "enable": {
                    "cases_windows": True,
                    "cases_accumulators": False,
                    "cases_lags": True,
                    "windows": False,
                },
                "windows": [4],
                "lags": 1,
            },
        )
        with tempfile.TemporaryDirectory() as output_dir:
            X, y, sidecar = create_new_features(
                contextual,
                "val",
                config,
                output_dir,
                target_start=split,
            )
            names = pd.read_csv(Path(output_dir) / "feature_dictionary.csv")["Feature"].tolist()

        self.assertEqual(pd.Timestamp(sidecar["DATE"][0]), split)
        self.assertEqual(y[0], 5.0)
        self.assertEqual(X[0, names.index("CASES_MM_4")], 2.5)
        self.assertEqual(X[0, names.index("CASES_LAG_1")], 3.0)

    def test_caseonly_is_filtered_to_full_support(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"])
        units = np.array(["A", "A", "B"])
        y = np.array([1.0, 2.0, 9.0])
        X = np.arange(6, dtype=float).reshape(3, 2)
        sidecar = {"DATE": dates.to_numpy(), "ID_UNIDADE": units, "Y_TRUE": y}

        with tempfile.TemporaryDirectory() as reference_dir:
            reference = {
                "test": {
                    "DATE": dates[:2].to_numpy(),
                    "ID_UNIDADE": units[:2],
                    "Y_TRUE": y[:2],
                }
            }
            with open(Path(reference_dir) / "dataset_ids.pickle", "wb") as file:
                import pickle
                pickle.dump(reference, file)

            X_filtered, y_filtered, ids_filtered = filter_to_reference_support(
                X, y, sidecar, reference_dir, "test"
            )

        np.testing.assert_array_equal(X_filtered, X[:2])
        np.testing.assert_array_equal(y_filtered, y[:2])
        np.testing.assert_array_equal(ids_filtered["ID_UNIDADE"], units[:2])

    def test_date_aggregation_preserves_between_unit_covariance(self):
        df = pd.DataFrame({
            "DATE": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "y_true": [1, 2],
            "y_pred_mean": [5, 5],
            "y_pred_std": [np.sqrt(50), np.sqrt(50)],
            "y_pred_seed_1": [0, 0],
            "y_pred_seed_2": [10, 10],
        })
        result = aggregate_by_date(df)
        row = result.loc[result["DATE"] == pd.Timestamp("2020-01-01")].iloc[0]
        self.assertEqual(row["y_pred_mean"], 10.0)
        self.assertAlmostEqual(row["y_pred_std"], np.std([0, 20], ddof=1))


if __name__ == "__main__":
    unittest.main()
