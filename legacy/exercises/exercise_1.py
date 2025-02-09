# Plot LST with neighborhoods

from netCDF4 import Dataset
import os, ogr
import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import numpy as np


def lat_lon_reproj(g16_data_file):
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

shapefile_path = "/run/media/metatron/Yggdrasil/python/arboseer/data/shapes/RJ_Municipios_2022/RJ_Municipios_2022"

##################
# create dummy figure for reading shapefile and setting boundaries
##################

fig1 = plt.figure()
m1 = Basemap()
shp = m1.readshapefile(shapefile_path, "shapefile")
plt.close(fig1)  # close dummy figure
m1 = ()

##################
# plotting actual shapefile
##################

#bbox = [shp[2][0], shp[2][1], shp[3][0], shp[3][1]]
bbox = [-44.98, -23.79, -40.92, -20.27]

fig, ax = plt.subplots(figsize=(14, 8))
m = Basemap(
    llcrnrlon=bbox[0],
    llcrnrlat=bbox[1],
    urcrnrlon=bbox[2],
    urcrnrlat=bbox[3],
    resolution="i",
    projection="cyl",
)
m.readshapefile(shapefile_path, "shapefile")
m.drawmapboundary(fill_color="#bdd5d5")
parallels = np.linspace(bbox[1], bbox[3], 5)  # latitudes
m.drawparallels(parallels, labels=[True, False, False, False], fontsize=12)
meridians = np.linspace(bbox[0], bbox[2], 5)  # longitudes
m.drawmeridians(meridians, labels=[False, False, False, True], fontsize=12)

##################
# finding boroughs and isolating them for shading in colors
##################

# BAIRROS #

##################
# GOES-16 LST section
##################

g16_data_file = "data/raw/lst/20230116/OR_ABI-L2-LSTF-M6_G16_s20230152100206_e20230152109514_c20230152111229.nc"
lon, lat = lat_lon_reproj(g16_data_file)

data = np.load('data/processed/lst/20230116.npz')['avg_ds']

##################
# clipping lat/lon/data vectors to shapefile bounds
##################
# lower-left corner
llcrnr_loc_x = np.ma.argmin(
    np.ma.min(
        np.abs(np.ma.subtract(lon, bbox[0])) + np.abs(np.ma.subtract(lat, bbox[1])), 0
    )
)
llcrnr_loc = (
    np.ma.argmin(
        np.abs(np.ma.subtract(lon, bbox[0])) + np.abs(np.ma.subtract(lat, bbox[1])), 0
    )
)[llcrnr_loc_x]
# upper-left corner
ulcrnr_loc_x = np.ma.argmin(
    np.ma.min(
        np.abs(np.ma.subtract(lon, bbox[0])) + np.abs(np.ma.subtract(lat, bbox[3])), 0
    )
)
ulcrnr_loc = (
    np.ma.argmin(
        np.abs(np.ma.subtract(lon, bbox[0])) + np.abs(np.ma.subtract(lat, bbox[3])), 0
    )
)[ulcrnr_loc_x]
# lower-right corner
lrcrnr_loc_x = np.ma.argmin(
    np.ma.min(
        np.abs(np.ma.subtract(lon, bbox[2])) + np.abs(np.ma.subtract(lat, bbox[1])), 0
    )
)
lrcrnr_loc = (
    np.ma.argmin(
        np.abs(np.ma.subtract(lon, bbox[2])) + np.abs(np.ma.subtract(lat, bbox[1])), 0
    )
)[lrcrnr_loc_x]
# upper-right corner
urcrnr_loc_x = np.ma.argmin(
    np.ma.min(
        np.abs(np.ma.subtract(lon, bbox[2])) + np.abs(np.ma.subtract(lat, bbox[3])), 0
    )
)
urcrnr_loc = (
    np.ma.argmin(
        np.abs(np.ma.subtract(lon, bbox[2])) + np.abs(np.ma.subtract(lat, bbox[3])), 0
    )
)[urcrnr_loc_x]

x_bounds = [llcrnr_loc_x, ulcrnr_loc_x, lrcrnr_loc_x, urcrnr_loc_x]
y_bounds = [llcrnr_loc, ulcrnr_loc, lrcrnr_loc, urcrnr_loc]

# setting bounds for new clipped lat/lon/data vectors
plot_bounds = [
    np.min(x_bounds),
    np.min(y_bounds),
    np.max(x_bounds) + 1,
    np.max(y_bounds) + 1,
]

# new clipped data vectors
lat_clip = lat[plot_bounds[1] : plot_bounds[3], plot_bounds[0] : plot_bounds[2]]
lon_clip = lon[plot_bounds[1] : plot_bounds[3], plot_bounds[0] : plot_bounds[2]]
dat_clip = data[plot_bounds[1] : plot_bounds[3], plot_bounds[0] : plot_bounds[2]]

# plot the new data only in the clipped region
#m.pcolormesh(
#    lon_clip, lat_clip, dat_clip, latlon=True, zorder=999, alpha=0.95, cmap="jet"
#)  # plotting actual LST data
#cb = m.colorbar()#
#plt.rc("text", usetex=True)
#plt.title("LST", fontsize=14)#
#plt.savefig(
#    "data/rj.png", dpi=200, facecolor=[252 / 255, 252 / 255, 252 / 255]
#)  # uncomment to save figure
#plt.show()
#

target_lat = -22.90223
target_lon = -43.27725
# Calculate the absolute difference between target and all points
lat_diff = np.abs(lat - target_lat)
lon_diff = np.abs(lon - target_lon)
# Combine the differences into a single distance metric
# Assuming lat/lon arrays are 2D and of the same shape as data
total_diff = np.sqrt(lat_diff**2 + lon_diff**2)#
# Find the index of the minimum difference
min_diff_idx = np.unravel_index(np.argmin(total_diff), total_diff.shape)#
# Retrieve the precipitation data for the closest point
precipitation_closest = data[min_diff_idx]
print(f"Closest value: {precipitation_closest}")
