# Training: Python and GOES-R Imagery: Read 500 m faster
 
# Required modules
from netCDF4 import Dataset                # Read / Write NetCDF4 files
import matplotlib.pyplot as plt            # Plotting library
import numpy as np                         # Scientific computing with Python
import cartopy, cartopy.crs as ccrs        # Plot maps
import cartopy.io.shapereader as shpreader # Import shapefiles
from datetime import datetime, timedelta   # Library to convert julian day to dd-mm-yyyy
 
# Open the GOES-R image
file = Dataset("data/TESTE/OR_ABI-L2-LSTF-M6_G16_s20230721300208_e20230721309515_c20230721311336.nc")

 
# Get the image resolution
band_resolution_km = getattr(file, 'spatial_resolution')
band_resolution_km = float(band_resolution_km[:band_resolution_km.find("km")])
 
# Desired visualization extent [min_lon, max_lon, min_lat, max_lat]
min_lon, max_lon, min_lat, max_lat = -44.88, -40.95, -23.36, -20.76
extent = [min_lon, min_lat, max_lon, max_lat]
 
# Read the GOES-R lat lons as arrays (files created previously)
lons = np.loadtxt('data/g16_lons_10km.txt')
lats = np.loadtxt('data/g16_lats_10km.txt')
ref_grid_resolution_km = 10
 
# Calculate the lat lon pairs indexes for the desired extent
idx_pair_1 = abs(lats-extent[1])+abs(lons-extent[0])
max_lat_idx,min_lon_idx = np.unravel_index(idx_pair_1.argmin(),idx_pair_1.shape)
idx_pair_2 = abs(lats-extent[3])+abs(lons-extent[2])
min_lat_idx,max_lon_idx = np.unravel_index(idx_pair_2.argmin(),idx_pair_2.shape)
 
# Adapt the reference indexes for the current file resolution
min_lat_idx = min_lat_idx * int(ref_grid_resolution_km/band_resolution_km)
min_lon_idx = min_lon_idx * int(ref_grid_resolution_km/band_resolution_km)
max_lat_idx = max_lat_idx * int(ref_grid_resolution_km/band_resolution_km)
max_lon_idx = max_lon_idx * int(ref_grid_resolution_km/band_resolution_km)
 
# The projection x and y coordinates equals the scanning angle (in radians) multiplied by the satellite height
sat_h = file.variables['goes_imager_projection'].perspective_point_height
x = file.variables['x'][min_lon_idx:max_lon_idx] * sat_h
y = file.variables['y'][min_lat_idx:max_lat_idx] * sat_h
 
# Get the pixel values
data = file.variables['LST'][min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx][::1,::1]
 
# Choose the plot size (width x height, in inches)
plt.figure(figsize=(7,7))
 
# Use the Geostationary projection in cartopy
ax = plt.axes(projection=ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0))
img_extent = (x.min(), x.max(), y.min(), y.max())
 
# Add a shapefile
shapefile = list(shpreader.Reader('/run/media/metatron/Yggdrasil/python/arboseer/data/shapes/RJ_Municipios_2022/RJ_Municipios_2022.shp').geometries())
ax.add_geometries(shapefile, ccrs.PlateCarree(), edgecolor='gold',facecolor='none', linewidth=0.3)
 
# Add coastlines, borders and gridlines
ax.coastlines(resolution='10m', color='white', linewidth=0.8)
#ax.add_feature(cartopy.feature.BORDERS, edgecolor='gray', linewidth=1.0)
ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
 
# Inserting City Labels
city_lons = [-43.1729, -43.1014]
city_lats = [-22.9068, -22.8807]
labels = ['Rio de Janeiro', 'Niteroi']
x_offsets = [0.1,0.1]
y_offsets = [0,0]
 
ax.plot(city_lons, city_lats, 'bo', color='cyan', markersize=5, transform=ccrs.Geodetic())
 
for label, xpt, ypt, x_offset, y_offset in zip(labels, city_lons, city_lats, x_offsets, y_offsets):
    ax.text(xpt+x_offset , ypt+y_offset, label, fontsize=12, fontweight='bold', zorder=8, color='black', transform=ccrs.Geodetic())
 
# Plot the image
img = ax.imshow(data, vmin=0.0, vmax=0.7, extent=img_extent, origin='upper', cmap='jet')
 
# Getting the file date
add_seconds = int(file.variables['time_bounds'][0])
date = datetime(2000,1,1,12) + timedelta(seconds=add_seconds)
date = date.strftime('%d %B %Y %H:%M UTC')
 
# Add a title
plt.title('GOES-16 Band 02 (500m)', fontweight='bold', fontsize=10, loc='left')
plt.title('Sub Region \n' + date, fontsize=10, loc='right')
 
# Save the image
plt.savefig('Image_08.png')
 
# Show the image
plt.show()