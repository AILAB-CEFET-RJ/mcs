# Plot quality data from LST

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import numpy as np
from datetime import datetime
import cartopy, cartopy.crs as ccrs        # Plot maps
import cartopy.io.shapereader as shpreader # Import shapefiles
from src.utils.download_goes_prod import download_goes_prod
from matplotlib.colors import ListedColormap

# Base image file
base_file_path = "data/raw/ADP/OR_ABI-L2-CMIPF-M6C13_G16_s20191981200396_e20191981210116_c20191981210189.nc"
file = Dataset(base_file_path)
data = file.variables['CMI'][:] - 273.15  # Convert to Celsius
longitude_of_projection_origin = file.variables['goes_imager_projection'].longitude_of_projection_origin
perspective_point_height = file.variables['goes_imager_projection'].perspective_point_height
xmin = file.variables['x'][:].min() * perspective_point_height
xmax = file.variables['x'][:].max() * perspective_point_height
ymin = file.variables['y'][:].min() * perspective_point_height
ymax = file.variables['y'][:].max() * perspective_point_height
#img_extent = (xmin, xmax, ymin, ymax)

# Rio de Janeiro extent
loni = -44.98
lonf = -23.79
lati = -40.92
latf = -20.27

img_extent = [loni, lonf, lati, latf]

# Directory containing ADP files
lst_dir = "data/raw/lst/20230101"
processed_dir = "data/processed/lst/timelapse"
os.makedirs(processed_dir, exist_ok=True)  # Ensure output directory exists

# List and sort all NetCDF files in the ADP directory
lst_files = sorted([f for f in os.listdir(lst_dir) if f.endswith('.nc')])

# Incremental counter for output filenames
n = 1

# Cores
colors = ['green', 'yellow', 'red', 'blue', 'black']
cmap = ListedColormap(colors)

for lst_file in lst_files:
    # Open the file
    file_adp = Dataset(os.path.join(lst_dir, lst_file))
    
    lst = file_adp.variables['DQF'][:]
       
    # Plotting setup
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0))
    ax.coastlines(resolution='10m', color='white', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.8)
    ax.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
    
    # Plot base image
    base_img = ax.imshow(data, vmin=-80, vmax=40, origin='upper', extent=img_extent, cmap='Greys')
    
    # Plot additional data layers
    #ax.imshow(lst, extent=img_extent, cmap='jet', alpha=1, origin='upper')
    ax.imshow(lst, extent=img_extent, cmap=cmap, alpha=1, origin='upper')
    
    # Title and save
    date = (datetime.strptime(file_adp.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ'))
    #plt.title(f'{date}', fontsize=10)
    plt.savefig(os.path.join(processed_dir, f"adp_{n:03d}.png"))
    plt.close()  # Close the plot to free memory
    
    #print(n)
    n += 1  # Increment file counter