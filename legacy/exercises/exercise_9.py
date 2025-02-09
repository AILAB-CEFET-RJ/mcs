# Testing Lat/Lon to X/Y coordinate function

import numpy as np
from netCDF4 import Dataset
def reproject_lats_lons(g16_data_file):
    # designate dataset
    g16nc = Dataset(g16_data_file, "r")
    var_names = [ii for ii in g16nc.variables]
    var_name = var_names[0]

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

    r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (2.0 * a_var)

    s_x = r_s * np.cos(lat_rad) * np.cos(lon_rad)
    s_y = -r_s * np.sin(lat_rad)
    s_z = r_s * np.cos(lat_rad) * np.sin(lon_rad)

    # latitude and longitude projection for plotting data on traditional lat/lon maps
    lat = (180.0 / np.pi) * (
        np.arctan(
            ((r_eq * r_eq) / (r_pol * r_pol))
            * ((s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y))))
        )
    )
    lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    return lon, lat

def find_closest_lat_lon(lat, lon, lats, lons):
    lat = float(lat)  # Ensure lat is a float
    lon = float(lon)  # Ensure lon is a float
    
    # Calculate the absolute difference and find the index of the smallest difference
    lat_diff = np.abs(lats - lat)
    lon_diff = np.abs(lons - lon)
    combined_diff = lat_diff + lon_diff
    return np.unravel_index(np.argmin(combined_diff), lats.shape)

# GOES-16 Product reference file. Can be any file of the desired product
reference_file = "data/raw/lst/OR_ABI-L2-LSTF-M6_G16_s20230010000206_e20230010009514_c20230010011292.nc"

# Reproject the lons and lats using the projection of the satelite in the file
lons, lats = reproject_lats_lons(reference_file)    

# Define de desired lat/lon
lat = -22.86
lon = -43.41

# Find the closet coordinates of the lat/lon on the array
x, y = find_closest_lat_lon(lat, lon, lats, lons)

print(f"{x},{y}")