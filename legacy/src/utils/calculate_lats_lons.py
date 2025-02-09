from netCDF4 import Dataset                # Read / Write NetCDF4 files
from pyproj import Proj                    # Cartographic projections and coordinate transformations library
import numpy as np                         # Scientific computing with Python
import logging
import argparse
import os
from download_goes_prod import download_goes_prod
 
def calculate_lats_lons(product, path):   
    try:
        file_path = f"{path}/{download_goes_prod('202312010000', product, path)}.nc"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file}' does not exist.")
            
        if not file_path.endswith('.nc'):
            raise ValueError(f"The file '{file}' is not a .nc file.")
                        
        # Any GOES netCDF4 file
        file = Dataset(file_path)
        os.makedirs(path, exist_ok=True)
        logging.info("Converting G16 coordinates to lons and lats...")

        sat_h = file.variables['goes_imager_projection'].perspective_point_height
        sat_lon = file.variables['goes_imager_projection'].longitude_of_projection_origin
        sat_sweep = file.variables['goes_imager_projection'].sweep_angle_axis

        # The projection x and y coordinates equals
        # the scanning angle (in radians) multiplied by the satellite height (http://proj4.org/projections/geos.html)
        X = file.variables['x'][:][::5] * sat_h
        Y = file.variables['y'][:][::5] * sat_h

        # map object with pyproj
        p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep, a=6378137.0)
        # Convert map points to latitude and longitude with the magic provided by Pyproj
        XX, YY = np.meshgrid(X, Y)
        lons, lats = p(XX, YY, inverse=True)
        
        # Pixels outside the globe as -9999
        mask = (lons == lons[0][0])
        lons[mask] = -9999
        lats[mask] = -9999

        coords = [lats, lons]
        np.savez(f'{path}/coords_{product}.npz', lats=lats, lons=lons)
        np.savetxt('data/processed/g16_lons_10km.txt', lons, fmt='%.5f')
        np.savetxt('data/processed/g16_lats_10km.txt', lats, fmt='%.5f')

        os.remove(file_path)

        logging.info("Saving lat long arrays for later use...")
    except Exception as e:
        logging.error(f"Failed to process file '{file_path}': {e}")        

def main():
    parser = argparse.ArgumentParser(description="Calculate Lat/Lon arrays for GOES-16 files")
    parser.add_argument("--product", dest="product", help="Product for which the coordinates will be generated", default="ABI-L2-RRQPEF")
    parser.add_argument("--path", dest="path", help="Destination path, default is data/raw", default="data/raw")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    
    
    calculate_lats_lons(args.product, args.path)

if __name__ == "__main__":
    main()