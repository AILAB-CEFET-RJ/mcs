from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_handling.stations.fuse_weather_sources_calibrated import validate_era5_units
from data_handling.stations.interpolate_to_grid import interpolate_channel


def test_minimum_station_count_masks_cell():
    values = np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32)
    weights = np.asarray([[[1.0, 0.5, 0.25]]], dtype=np.float32)
    valid, _ = interpolate_channel(np.arange(1), values, weights, min_stations=3)
    masked, _ = interpolate_channel(np.arange(1), values, weights, min_stations=4)
    assert np.isfinite(valid[0, 0, 0])
    assert np.isnan(masked[0, 0, 0])


def test_nearest_uses_largest_inverse_distance_weight():
    values = np.asarray([[10.0, 20.0]], dtype=np.float32)
    weights = np.asarray([[[0.2, 1.0]]], dtype=np.float32)
    result, _ = interpolate_channel(
        np.arange(1), values, weights, method="nearest", min_stations=1
    )
    assert result[0, 0, 0] == 20.0


def test_unit_guard_rejects_rain_in_metres():
    fake = {"channel_units": np.asarray(["degC", "m"])}
    with pytest.raises(ValueError, match="RAIN must be mm"):
        validate_era5_units(fake, ["TEM_AVG", "RAIN"], False)
