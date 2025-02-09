import os                                       # Miscellaneous operating system interfaces
import numpy as np                              # Import the Numpy package
import colorsys                                 # To make convertion of colormaps
import math                                     # Mathematical functions
from osgeo import osr                           # Python bindings for GDAL
from osgeo import gdal                          # Python bindings for GDAL
from netCDF4 import Dataset                     # Read / Write NetCDF4 files
from datetime import datetime                   # Python datetime library
import matplotlib.pyplot as plt                 # Plotting library
import cartopy, cartopy.crs as ccrs             # Plot maps
import cartopy.io.shapereader as shpreader      # Import shapefiles
gdal.PushErrorHandler('CPLQuietErrorHandler')   # Ignore GDAL warnings


def reproject(file_name, ncfile, array, extent, undef):

    # Read the original file projection and configure the output projection
    source_prj = osr.SpatialReference()
    source_prj.ImportFromProj4(ncfile.GetProjectionRef())

    target_prj = osr.SpatialReference()
    target_prj.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

    # Reproject the data
    GeoT = ncfile.GetGeoTransform()
    driver = gdal.GetDriverByName('MEM')
    raw = driver.Create('raw', array.shape[0], array.shape[1], 1, gdal.GDT_Float32)
    raw.SetGeoTransform(GeoT)
    raw.GetRasterBand(1).WriteArray(array)

    # Define the parameters of the output file
    kwargs = {'format': 'netCDF', \
            'srcSRS': source_prj, \
            'dstSRS': target_prj, \
            'outputBounds': (extent[0], extent[3], extent[2], extent[1]), \
            'outputBoundsSRS': target_prj, \
            'outputType': gdal.GDT_Float32, \
            'srcNodata': undef, \
            'dstNodata': 'nan', \
            'resampleAlg': gdal.GRA_NearestNeighbour}

    # Write the reprojected file on disk
    gdal.Warp(file_name, raw, **kwargs)

def process_file(file_path):
    output = "data/processed/figs"; os.makedirs(output, exist_ok=True)

    # Rio de Janeiro
    loni = -44.98
    lonf = -23.79
    lati = -40.92
    latf = -20.27

    extent = [loni, lonf, lati, latf]

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    reference_file_path = 'data/raw/rrqpe/20230101/OR_ABI-L2-RRQPEF-M6_G16_s20223652100206_e20223652109514_c20223652110017.nc'
    var = 'RRQPE'

    #-----------------------------------------------------------------------------------------------------------
    # Load pre-calculated data
    data = np.load(file_path)['acum']
    #-----------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------
    # Open any file of the product to get the metadata
    img = gdal.Open(f'NETCDF:{reference_file_path}:{var}')
    metadata = img.GetMetadata()
    undef = float(metadata.get(f'{var}#_FillValue'))    
    dtime = metadata.get('NC_GLOBAL#time_coverage_start')
    #-----------------------------------------------------------------------------------------------------------    

    #-----------------------------------------------------------------------------------------------------------
    # Reproject the file
    reprojected_file_name = f'{output}/{file_name}_avg.nc'
    reproject(reprojected_file_name, img, data, extent, undef)
    #-----------------------------------------------------------------------------------------------------------
    # Open the reprojected GOES-R image
    reprojected_file = Dataset(reprojected_file_name)

    # Get the pixel values
    reprojected_data = reprojected_file.variables['Band1'][:]
    #-----------------------------------------------------------------------------------------------------------
    # Choose the plot size (width x height, in inches)
    plt.figure(figsize=(10,10))

    # Use the Geostationary projection in cartopy
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Define the image extent
    img_extent = [extent[0], extent[2], extent[1], extent[3]]

    # Plot the image
    img = ax.imshow(reprojected_data, vmin=1, vmax=100, cmap='viridis', origin='upper', extent=img_extent)

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
    plt.colorbar(img, label='LST (°C)', extend='both', orientation='horizontal', pad=0.05, fraction=0.05)

    # Extract the date
    date = (datetime.strptime(dtime, '%Y-%m-%dT%H:%M:%S.%fZ'))

    # Add a title
    plt.title(f'GOES-16 {var} ' + date.strftime('%Y-%m-%d %H:%M') + ' UTC', fontweight='bold', fontsize=10, loc='left')
    plt.title('Reg.: ' + str(extent) , fontsize=10, loc='right')
    #-----------------------------------------------------------------------------------------------------------
    # Save the image
    plt.savefig(f'{output}/{file_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)

input_path = "data/processed/test"
for file in os.listdir(input_path):      
    file_path = os.path.join(input_path, file)
    if os.path.isfile(file_path):
        process_file(file_path)    