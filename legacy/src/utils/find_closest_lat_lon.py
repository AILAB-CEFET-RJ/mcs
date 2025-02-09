# USE ONLY ON ARRAYS THAT HAVEN'T BEEN REPROJECTED

import numpy as np

def find_closest_lat_lon(lat, lon, lats, lons):
    lat = float(lat)  # Ensure lat is a float
    lon = float(lon)  # Ensure lon is a float
    
    # Calculate the absolute difference and find the index of the smallest difference
    lat_diff = np.abs(lats - lat)
    lon_diff = np.abs(lons - lon)
    combined_diff = lat_diff + lon_diff
    return np.unravel_index(np.argmin(combined_diff), lats.shape)