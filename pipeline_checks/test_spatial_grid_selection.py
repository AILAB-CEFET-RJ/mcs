import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

from arboseer.src.data_handling.spatial_grid.selection import (
    assign_points,
    build_grid,
    evaluate_candidate,
)
from arboseer.src.data_handling.spatial_grid.regrid_era5 import regrid


def test_metric_grid_assignment_and_train_only_metrics():
    territory = box(500_000, 7_450_000, 510_000, 7_460_000)
    grid = build_grid("synthetic", "EPSG:31983", 5, territory)
    assert (grid.rows, grid.cols) == (2, 2)

    # Convert projected cell centers to geographic input expected by CNES.
    inv = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)
    lon1, lat1 = inv.transform(502_500, 7_457_500)
    lon2, lat2 = inv.transform(507_500, 7_452_500)
    cnes = pd.DataFrame(
        {"CNES": ["1", "2", "3"], "LNG": [lon1, lon1, lon2], "LAT": [lat1, lat1, lat2]}
    )
    train = pd.DataFrame(
        {
            "ID_UNIDADE": ["1", "2"],
            "CASES": [3, 4],
            "DT_NOTIFIC": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    metrics, cells, masks = evaluate_candidate(grid, territory, cnes, train)
    assert metrics["cells_with_cnes"] == 2
    assert metrics["cells_with_train_cases"] == 1
    assert metrics["train_cases_total"] == 7
    assert metrics["cnes_per_cell_max"] == 2
    assert masks["train_active"].sum() == 1
    assert cells["cnes_count"].sum() == 3


def test_extent_edges_are_included():
    territory = box(500_000, 7_450_000, 510_000, 7_460_000)
    grid = build_grid("synthetic", "EPSG:31983", 5, territory)
    inv = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(510_000, 7_450_000)
    row, col, valid = assign_points(grid, np.array([lon]), np.array([lat]))
    assert valid[0]
    assert (row[0], col[0]) == (1, 1)


def test_era5_regridding_keeps_source_pixel_provenance(tmp_path):
    source_path = tmp_path / "source.npz"
    np.savez(
        source_path,
        era5_weekly=np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2),
        lat=np.array([-23.0, -22.0]),
        lon=np.array([-44.0, -43.0]),
    )
    cells = pd.DataFrame(
        {"row": [0], "col": [0], "latitude": [-22.5], "longitude": [-43.5]}
    )
    source = np.load(source_path)
    values, lat, lon, contributors, fallback, nearest_id = regrid(source, cells, "linear")
    assert values.shape == (1, 1, 1, 1)
    assert values.item() == 1.5
    assert set(contributors.ravel()) == {0, 1, 2, 3}
    assert lat.item() == -22.5 and lon.item() == -43.5
    assert not fallback.any()
    assert (nearest_id == -1).all()
