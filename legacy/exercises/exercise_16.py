import os                                       # Miscellaneous operating system interfaces
import numpy as np                              # Import the Numpy package
import colorsys                                 # To make convertion of colormaps
import math                                     # Mathematical functions
from osgeo import osr                           # Python bindings for GDAL
from osgeo import gdal                          # Python bindings for GDAL
from netCDF4 import Dataset                     # Read / Write NetCDF4 files
import matplotlib.pyplot as plt                 # Plotting library
import cartopy, cartopy.crs as ccrs             # Plot maps
import cartopy.io.shapereader as shpreader      # Import shapefiles
gdal.PushErrorHandler('CPLQuietErrorHandler')   # Ignore GDAL warnings
from datetime import datetime   # Library to convert julian day to dd-mm-yyyy
from matplotlib.colors import ListedColormap
import pandas as pd
from scipy.ndimage import binary_dilation

def find_closest_lat_lon(lat, lon, lats, lons):
    lat = float(lat)  # Ensure lat is a float
    lon = float(lon)  # Ensure lon is a float
    
    # Calculate the absolute difference and find the index of the smallest difference
    lat_diff = np.abs(lats - lat)
    lon_diff = np.abs(lons - lon)
    combined_diff = lat_diff + lon_diff
    return np.unravel_index(np.argmin(combined_diff), lats.shape)

def iterative_dilation(array, max_value):
    rows, cols = array.shape
    output_array = np.copy(array)

    # Executando várias iterações, uma para cada valor
    for current_value in range(1, max_value):
        # Criar uma cópia para evitar modificação durante a iteração
        temp_array = np.copy(output_array)
        
        for x in range(rows):
            for y in range(cols):
                if output_array[x, y] == current_value:
                    # Lista de posições adjacentes para atualizar
                    neighbors = [
                        (x-1, y),   # Acima
                        (x, y-1),   # Esquerda
                        (x-1, y-1), # Diagonal superior esquerda
                        (x+1, y-1), # Diagonal inferior esquerda
                        (x+1, y),   # Abaixo
                        (x, y+1),   # Direita
                        (x-1, y+1), # Diagonal superior direita
                        (x+1, y+1)  # Diagonal inferior direita
                    ]
                    
                    # Aplicar atualizações se as posições são válidas e o valor atual é 0
                    for nx, ny in neighbors:
                        if 0 <= nx < rows and 0 <= ny < cols and temp_array[nx, ny] == 0:
                            temp_array[nx, ny] = current_value + 1
        
        # Atualizar o output_array com as mudanças da iteração atual
        output_array = temp_array

    return output_array

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
    output = "data/processed/misc"; os.makedirs(output, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Rio de Janeiro extent
    loni = -44.98
    lonf = -23.79
    lati = -40.92
    latf = -20.27

    extent = [loni, lonf, lati, latf]
    lons, lats = reproject_lats_lons(file_path)
    df_stations = pd.read_csv('stations.csv')
    data = np.empty((1086,1086))    

    for index, row in df_stations.iterrows():
        lat = row['VL_LATITUDE']
        lon = row['VL_LONGITUDE']
        x, y = find_closest_lat_lon(lat, lon, lats, lons)
        data[x,y] = 1
    
    img = gdal.Open(f'NETCDF:{file_path}:LST')
    metadata = img.GetMetadata()
    undef = float(metadata.get('LST#_FillValue'))    
    dtime = metadata.get("NC_GLOBAL#time_coverage_start")

    #-----------------------------------------------------------------------------------------------------------
    # Reproject the file
    reprojected_file_name = f'{output}/{file_name}_avg.nc'
    reproject(reprojected_file_name, img, data, extent, undef)
    #-----------------------------------------------------------------------------------------------------------
    # Open the reprojected GOES-R image
    reprojected_file = Dataset(reprojected_file_name)

    # Get the pixel values
    reprojected_data = reprojected_file.variables['Band1'][:]

    #reprojected_data = iterative_dilation(reprojected_data, 3)

    #-----------------------------------------------------------------------------------------------------------
    # Choose the plot size (width x height, in inches)
    plt.figure(figsize=(10,10))

    # Use the Geostationary projection in cartopy
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Define the image extent
    img_extent = [extent[0], extent[2], extent[1], extent[3]]

    # Plot the image
    colors = ['white', 'red', 'orange', 'yellow']
    colors = ['white', "black"]
    cmap = ListedColormap(colors)
    img = ax.imshow(reprojected_data, cmap=cmap, origin='upper', extent=img_extent)

    # Add coastlines, borders and gridlines
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)

    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.STATES, edgecolor='black', linewidth=0.5)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    plt.xlim(extent[0], extent[2])
    plt.ylim(extent[1], extent[3])

    # Extract the date
    date = (datetime.strptime(dtime, '%Y-%m-%dT%H:%M:%S.%fZ'))

    #-----------------------------------------------------------------------------------------------------------
    # Save the image
    plt.savefig(f'{output}/{file_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)

process_file('data/raw/lst/20230101/OR_ABI-L2-LSTF-M6_G16_s20230010300206_e20230010309514_c20230010311285.nc')