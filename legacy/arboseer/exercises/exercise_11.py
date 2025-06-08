# Plot processed LST data for temporal series

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import os
import numpy as np
from datetime import datetime
import cartopy, cartopy.crs as ccrs             # Plot maps
import cartopy.io.shapereader as shpreader      # Import shapefiles
from osgeo import osr                           # Python bindings for GDAL
from osgeo import gdal                          # Python bindings for GDAL
gdal.PushErrorHandler('CPLQuietErrorHandler')   # Ignore GDAL warnings

def process_file(file_path):
    output = "data/processed/lst_ts/validation/"; os.makedirs(output, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Rio de Janeiro extent
    loni = -44.98
    lonf = -23.79
    lati = -40.92
    latf = -20.27

    extent = [loni, lonf, lati, latf]    
      
    lst = Dataset(file_path)['Band1'][:]

    #-----------------------------------------------------------------------------------------------------------
    # Choose the plot size (width x height, in inches)
    plt.figure(figsize=(10,10))

    # Use the Geostationary projection in cartopy
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Define the image extent
    img_extent = [extent[0], extent[2], extent[1], extent[3]]

    # Plot the image
    img = ax.imshow(lst, cmap='jet', origin='upper', extent=img_extent)

    # Add coastlines, borders and gridlines
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)

    shapefile = list(shpreader.Reader('data/shapes/BR_UF_2019.shp').geometries())
    ax.add_geometries(shapefile, ccrs.PlateCarree(), edgecolor='black',facecolor='none', linewidth=0.3)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    plt.xlim(extent[0], extent[2])
    plt.ylim(extent[1], extent[3])

    # Add a colorbar
    #plt.colorbar(img, label='LST (°C)', extend='both', orientation='horizontal', pad=0.05, fraction=0.05)

    # Extract the date

    # Add a title
    plt.title('GOES-16 LST - 202312310400')
    plt.title('Reg.: ' + str(extent) , fontsize=10, loc='right')
    #-----------------------------------------------------------------------------------------------------------
    # Save the image
    plt.savefig(f'{output}/{file_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)
    #plt.show()

process_file('data/processed/lst_ts/202312310100.nc')