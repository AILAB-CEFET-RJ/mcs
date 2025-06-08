# FOR USE BEFORE GDAL REPROJECTION

import numpy as np
from netCDF4 import Dataset

def reproject_lats_lons(g16_data_file):
    # designate dataset
    g16nc = Dataset(g16_data_file, "r")

    # GOES-R projection info and retrieving relevant constants
    proj_info = g16nc.variables["goes_imager_projection"]
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height + proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis

    # grid info
    lat_rad_1d = g16nc.variables["x"][:]
    lon_rad_1d = g16nc.variables["y"][:]

    # close file when finished
    g16nc.close()
    g16nc = None

    # create meshgrid filled with radian angles
    lat_rad, lon_rad = np.meshgrid(lat_rad_1d, lon_rad_1d)

    # lat/lon calc routine from satellite radian angle vectors

    lambda_0 = (lon_origin * np.pi) / 180.0

    a_var = np.power(np.sin(lat_rad), 2.0) + (
        np.power(np.cos(lat_rad), 2.0)
        * (
            np.power(np.cos(lon_rad), 2.0)
            + (((r_eq * r_eq) / (r_pol * r_pol)) * np.power(np.sin(lon_rad), 2.0))
        )
    )
    b_var = -2.0 * H * np.cos(lat_rad) * np.cos(lon_rad)
    c_var = (H**2.0) - (r_eq**2.0)

    discriminant = (b_var**2) - (4.0 * a_var * c_var)
    discriminant[discriminant < 0] = np.nan  # Ensure non-negative values

    r_s = (-1.0 * b_var - np.sqrt(discriminant)) / (2.0 * a_var)

    s_x = r_s * np.cos(lat_rad) * np.cos(lon_rad)
    s_y = -r_s * np.sin(lat_rad)
    s_z = r_s * np.cos(lat_rad) * np.sin(lon_rad)

    # latitude and longitude projection for plotting data on traditional lat/lon maps
    with np.errstate(invalid='ignore'):
        sqrt_term = np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y))
        invalid_mask = sqrt_term == 0

        sqrt_term[invalid_mask] = np.nan

        lat = (180.0 / np.pi) * (
            np.arctan(
                ((r_eq * r_eq) / (r_pol * r_pol))
                * ((s_z / sqrt_term))
            )
        )
        lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    lat[invalid_mask] = np.nan
    lon[invalid_mask] = np.nan

    return lon, lat