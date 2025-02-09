from pathlib import Path

import xarray as xr

from .GreatCircleDistance import great_circle_distance
from .Logger import logger

log = logger.get_logger(__name__)


def _get_lats_lons() -> tuple[list[float], list[float]]:
    era5_land_path = Path(__file__).parent / "ERA5Land" / "montly_data"
    if not era5_land_path.exists():
        log.error(f"ERA5Land path {era5_land_path} does not exist.")
        era5_land_path = Path(__file__).parent / "ERA5Land" / "monthly_data"
        if not era5_land_path.exists():
            log.error(f"ERA5Land path {era5_land_path} does not exist.")
            exit(1)

    era5land_nc_ds = xr.open_dataset(next(era5_land_path.iterdir()))
    latitudes = era5land_nc_ds.coords["latitude"].values
    longitudes = era5land_nc_ds.coords["longitude"].values
    log.info(f"latitudes: {latitudes}")
    log.info(f"longitudes: {longitudes}")
    return latitudes, longitudes


def get_nearest_ERA5Land(latitude: float, longitude: float) -> tuple[float, float]:
    min_distance = float("inf")
    nearest_station = None
    era5land_lats, era5land_lons = _get_lats_lons()
    for lat in era5land_lats:
        for lon in era5land_lons:
            distance = great_circle_distance.get_distance(latitude, longitude, lat, lon)
            if distance < min_distance:
                min_distance = distance
                nearest_station = (lat, lon)
    return nearest_station
