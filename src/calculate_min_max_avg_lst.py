import pandas as pd
import logging
import os
from datetime import datetime, timedelta
import argparse
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.download_lst_data import download_lst_data
from utils.reproject_nc_file import reproject

gdal.PushErrorHandler('CPLQuietErrorHandler')

count = 0

def download_lst(path, date):
    download_lst_data(date, f"{path}/{date}")

def process_file(file, date, lon, lat, count):
    extent = [-44.7930, -40.7635, -23.3702, -20.7634]  # TODO: Move to args
    loni, lonf, lati, latf = extent

    lst_variables = gdal.Open(f'NETCDF:{file}:LST')
    dqf_variables = gdal.Open(f'NETCDF:{file}:DQF')
    metadata = lst_variables.GetMetadata()
    scale = float(metadata.get('LST#scale_factor'))
    offset = float(metadata.get('LST#add_offset'))

    lst_data = lst_variables.ReadAsArray(0, 0, lst_variables.RasterXSize, lst_variables.RasterYSize).astype(float)
    dqf_data = dqf_variables.ReadAsArray(0, 0, dqf_variables.RasterXSize, dqf_variables.RasterYSize).astype(float)

    # Apply the scale, offset and convert to Celsius
    lst_data = (lst_data * scale + offset) - 273.15

    # Apply NaNs where the quality flag is greater than 1
    lst_data[dqf_data > 1] = np.nan

    # Reproject the data
    reprojected_file_name = f'data/processed/{date}_{count}_reprojected.nc'
    undef = float(metadata.get('LST#_FillValue'))
    reproject(reprojected_file_name, lst_variables, lst_data, extent, undef)

    # Open the reprojected file
    reprojected_file = gdal.Open(reprojected_file_name)
    reprojected_data = reprojected_file.ReadAsArray().astype(float)

    # Transform coordinates to lat/lon
    geo_transform = reprojected_file.GetGeoTransform()
    x_origin = geo_transform[0]
    y_origin = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = geo_transform[5]

    cols = reprojected_file.RasterXSize
    rows = reprojected_file.RasterYSize

    lats = y_origin + pixel_height * (0.5 + np.arange(rows))
    lons = x_origin + pixel_width * (0.5 + np.arange(cols))

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    df = pd.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten(),
        'LST': reprojected_data.flatten()
    })

    df = df[(df['latitude'] >= lati) & (df['latitude'] <= latf) & (df['longitude'] >= loni) & (df['longitude'] <= lonf)]
    df = df.dropna(subset=['LST'])
    df['date'] = date  # Add the date column

    os.remove(reprojected_file_name)

    return df[['date', 'LST', 'latitude', 'longitude']]

def create_dataset_from_files(path, date):
    lstpath = f'{path}/{date}'
    files = [os.path.join(lstpath, f) for f in os.listdir(lstpath) if f.endswith('.nc')]

    data_list = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, date, None, None, idx): file for idx, file in enumerate(files)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                data_list.append(future.result())
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")

    if data_list:
        combined_df = pd.concat(data_list, ignore_index=True)
    else:
        combined_df = pd.DataFrame(columns=['date', 'LST', 'latitude', 'longitude'])

    return combined_df

def calculate_min_max_avg_lst(date, lstpath):
    #download_lst(lstpath, date)
    dataset = create_dataset_from_files(lstpath, date)
    aggregated_df = dataset.groupby(['latitude', 'longitude']).agg(
        LST_MAX=('LST', 'max'),
        LST_MIN=('LST', 'min'),
        LST_AVG=('LST', 'mean')
    ).reset_index()
    aggregated_df['date'] = date
    aggregated_list.append(aggregated_df)

aggregated_list = []
def main():
    parser = argparse.ArgumentParser(description="Calculate LST data for a date range, saving NP arrays")
    parser.add_argument("start", help="Date start in yyyymmdd format")
    parser.add_argument("end", help="Date end in yyyymmdd format")
    parser.add_argument("lstpath", help="Path where lst files will be downloaded, default is data/raw", default="data/raw")
    parser.add_argument("destpath", help="Destination path, default is data/raw", default="data/raw")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    start_date = datetime.strptime(args.start, '%Y%m%d')
    end_date = datetime.strptime(args.end, '%Y%m%d')

    current_date = start_date

    while current_date <= end_date:
        logging.info(f"Calculating LST for {current_date.strftime('%Y%m%d')}")

        calculate_min_max_avg_lst(current_date.strftime('%Y%m%d'), args.lstpath)
        current_date += timedelta(days=1)

    final_df = pd.concat(aggregated_list, ignore_index=True)
    final_df.to_parquet(args.destpath+'/lst.parquet', index=False)

if __name__ == "__main__":
    main()
