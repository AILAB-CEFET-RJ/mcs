# Find an extent of a target size

# Globals
#-----------------------------------------------------------------------------------------------------------
import importlib.util
import os
import sys
import numpy as np
from datetime import datetime
from osgeo import gdal
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy, cartopy.crs as ccrs
gdal.PushErrorHandler('CPLQuietErrorHandler') 

def dynamic_import(module_name, function_name, directory="src"):
    """Dynamically imports a function or class from a given module."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, base_path)
    
    module_path = os.path.join(base_path, directory, *module_name.split('.'))
    spec = importlib.util.spec_from_file_location(module_name, f"{module_path}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return getattr(module, function_name)

reproject = dynamic_import("utils.reproject_nc_file", "reproject")

from osgeo import gdal
from netCDF4 import Dataset
gdal.PushErrorHandler('CPLQuietErrorHandler')
#-----------------------------------------------------------------------------------------------------------

# Iterate until target shape is met
#-----------------------------------------------------------------------------------------------------------

# Initial extent and target shape for Rio de Janeiro
loni, lonf, lati, latf = -50.60, -22.60, -44.00, -16.00
loni, lonf, lati, latf = -44.98, -23.79, -40.92, -20.27
extent = [loni, lonf, lati, latf]
t_x = 64 
t_y = 64
target_shape = (t_x, t_y)

# Increment values for adjusting extent
lon_increment = 0.1  # Adjust longitude by 0.1 degrees each time
lat_increment = 0.1  # Adjust latitude by 0.1 degrees each time

# Load the NetCDF file and LST array
helper_file_path = 'data/raw/lst/20230101/OR_ABI-L2-LSTF-M6_G16_s20230012000206_e20230012009514_c20230012011222.nc'
lst_variables = gdal.Open(f'NETCDF:{helper_file_path}:LST')
metadata = lst_variables.GetMetadata()
undef = float(metadata.get('LST#_FillValue'))
array = Dataset(helper_file_path)['LST'][:]
target_met = True
while not target_met:
    reproject('output.nc', lst_variables, array, extent, undef)

    with Dataset('output.nc') as reprojected_dataset:
        reprojected_shape = reprojected_dataset['Band1'][:].shape
        print(reprojected_shape)
        if reprojected_shape == target_shape:
            target_met = True
        else:
            # Adjust longitude (west) by a fixed amount
            if reprojected_shape[1] != t_x:
                extent[0] -= lat_increment  # Adjust the west boundary
            else:
                if reprojected_shape[0] != t_y:
                    extent[1] -= lon_increment  # Adjust the north boundary               

reproject('output.nc', lst_variables, array, extent, undef)
print(f'Extent found: {extent}')

#-----------------------------------------------------------------------------------------------------------

# Plot figure
#-----------------------------------------------------------------------------------------------------------
file = Dataset('output.nc')
data = file.variables['Band1'][:]
data = np.full(data.shape, np.nan)


porcentagem = 0.1
num_valores = int(data.size * porcentagem)
indices = np.random.choice(data.size, num_valores, replace=False)
rows, cols = np.unravel_index(indices, data.shape)
data[rows, cols] = -np.inf
mask_inf = data == -np.inf

plt.figure(figsize=(10,10))

# Use the Geostationary projection in cartopy
ax = plt.axes(projection=ccrs.PlateCarree())

# Define the image extent
img_extent = [extent[0], extent[2], extent[1], extent[3]]

# Plot the image
#img = ax.imshow(data, cmap='jet', origin='upper', extent=img_extent)
img = ax.imshow(np.ma.masked_where(mask_inf, data), cmap='jet', origin='upper', extent=img_extent)
#ax.imshow(np.ma.masked_where(~mask_inf, mask_inf), cmap='gray', origin='upper', alpha=1, extent=img_extent)

# Add coastlines, borders and gridlines
ax.coastlines(resolution='10m', color='black', linewidth=0.8)
ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
gl.top_labels = False
gl.right_labels = False

plt.xlim(extent[0], extent[2])
plt.ylim(extent[1], extent[3])
plt.tight_layout()

# Add a title
plt.title('Reg.: ' + str(extent) , fontsize=10, loc='right')
#-----------------------------------------------------------------------------------------------------------

# Show the image
plt.show()