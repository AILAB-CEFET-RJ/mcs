# Globals
#-----------------------------------------------------------------------------------------------------------
from datetime import datetime
from datetime import timedelta
import os
import logging
import numpy as np
from src.utils.download_lst_data import download_lst_data
from src.utils.reproject_nc_file import reproject
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler') 
import re
from netCDF4 import Dataset
import random
from src.utils.download_inmet_data import retrieve_data
from src.utils.find_closest_lat_lon import find_closest_lat_lon
from src.utils.reproject_lats_lons import reproject_lats_lons
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

logging.basicConfig(level=getattr(logging, "INFO"), format="%(asctime)s - %(levelname)s - %(message)s")    

start_date = '20230101'
end_date = '20231231'
data_path = 'data'
raw_path = data_path+'/raw'
raw_lst_path = raw_path+'/lst_ts'
raw_inmet_path = raw_path+'/inmet_ts'
processed_path = data_path+'/processed'
processed_lst_path = processed_path+'/lst_ts'
processed_lst_path_gaps = processed_path+'/lst_ts_gaps'
processed_inmet_path = processed_path+'/inmet_ts'
processed_misc_path = processed_lst_path+'/misc'
stations_csv_path = 'stations.csv'

# Rio de Janeiro extent
loni = -44.98
lonf = -23.79
lati = -40.92
latf = -20.27

extent = [loni, lonf, lati, latf]

#-----------------------------------------------------------------------------------------------------------

# First, download the LST data
#-----------------------------------------------------------------------------------------------------------
def download_goes():
    start_date_formated = datetime.strptime(start_date, '%Y%m%d')
    end_date_formated = datetime.strptime(end_date, '%Y%m%d')
    current_date = start_date_formated
    os.makedirs(raw_lst_path, exist_ok=True)

    while current_date <= end_date_formated:
        logging.info(f"Downloading LST data for {current_date.strftime('%Y%m%d')}")
        download_lst_data(current_date.strftime('%Y%m%d'), f"{raw_lst_path}")
        current_date += timedelta(days=1)

    logging.info(f"Done downloading LST data")    
#-----------------------------------------------------------------------------------------------------------

# Now, reproject the data and create the array mask for places that never have value
#-----------------------------------------------------------------------------------------------------------
def reproject_files():
    os.makedirs(processed_lst_path, exist_ok=True)
    os.makedirs(processed_misc_path, exist_ok=True)
    mask_array = None

    files = os.listdir(raw_lst_path)
    files.sort(key=lambda x: datetime.strptime(re.search(r'_c(\d{13})', x).group(1), "%Y%j%H%M%S"))    

    for raw_lst_file in files:
        raw_lst_file_path = os.path.join(raw_lst_path, raw_lst_file)
        if os.path.isfile(raw_lst_file_path):
            # Load the metadata
            logging.info(f"Reprojecting file: {raw_lst_file}")
            lst_variables = gdal.Open(f'NETCDF:{raw_lst_file_path}:LST')
            dqf_variables = gdal.Open(f'NETCDF:{raw_lst_file_path}:DQF')
            metadata = lst_variables.GetMetadata()
            scale = float(metadata.get('LST#scale_factor'))
            offset = float(metadata.get('LST#add_offset'))
            undef = float(metadata.get('LST#_FillValue'))
            dtime = metadata.get('NC_GLOBAL#time_coverage_start')
            data_formatada = datetime.strptime(dtime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y%m%d%H%M") 

            # Load the data
            lst_data = lst_variables.ReadAsArray(0, 0, lst_variables.RasterXSize, lst_variables.RasterYSize).astype(float)
            dqf_data = dqf_variables.ReadAsArray(0, 0, dqf_variables.RasterXSize, dqf_variables.RasterYSize).astype(float)

            # Apply the scale, offset and convert to celsius
            lst_data = (lst_data * scale + offset) - 273.15

            # Apply NaN's where the quality flag is greater than 1
            lst_data[dqf_data > 1] = np.nan

            if mask_array is None:
                mask_array = np.zeros_like(lst_data, dtype=int)
           
            reprojected_filename = f'{processed_lst_path}/{data_formatada}.nc'
            reproject(reprojected_filename, lst_variables, lst_data, extent, undef)

            mask_array[~np.isnan(lst_data)] = 1
    
    logging.info(f"Done reprojecting LST files")

    reprojected_filename = f'{processed_misc_path}/mask.nc'
    reproject(reprojected_filename, lst_variables, mask_array, extent, undef)

    logging.info(f"Saved LST info {processed_misc_path}/mask.nc")
#-----------------------------------------------------------------------------------------------------------        

# After reprojecting the files, we will introduce gaps
#-----------------------------------------------------------------------------------------------------------
def generate_gaps():
    def introduzir_gaps(data, mask):
        new_gap = False
        area_total = np.prod(data.shape) - np.sum(mask)
        preenchimento_atual = np.sum(~np.isnan(data) & (mask == 1)) / area_total
        min_preenchimento = 0.2  # example minimum filling percentage
        prob_gap = 0.5  # probability of introducing a gap
        max_gaps = 3  # maximum number of gaps
        max_gap_size = 5  # maximum size of each gap
        prob_continuar = 0.3  # probability to continue the gap
        min_gap_duration = 1  # minimum gap duration
        max_gap_duration = 3  # maximum gap duration

        gaps_info = []
        if preenchimento_atual > min_preenchimento and random.random() < prob_gap:
            num_gaps = random.randint(1, max_gaps)
            for _ in range(num_gaps):
                gap_tamanho = random.randint(1, max_gap_size)
                gap_duracao = 1
                if random.random() < prob_continuar:
                    gap_duracao = random.randint(min_gap_duration, max_gap_duration)

                pos_valida = False
                for _ in range(100):
                    x = random.randint(0, data.shape[0] - gap_tamanho)
                    y = random.randint(0, data.shape[1] - gap_tamanho)
                    selected_area = data[x:x+gap_tamanho, y:y+gap_tamanho]
                    selected_mask_area = mask[x:x+gap_tamanho, y:y+gap_tamanho]
                    
                    if np.all(selected_mask_area == 1) and not np.any(np.isnan(selected_area)):
                        pos_valida = True
                        break

                if not pos_valida:
                    continue

                data[x:x+gap_tamanho, y:y+gap_tamanho] = np.nan
                gaps_info.append((x, y, gap_tamanho, gap_duracao))
                new_gap = True

        return data, gaps_info, new_gap

    os.makedirs(processed_lst_path_gaps, exist_ok=True)

    files = [f for f in os.listdir(processed_lst_path) if os.path.isfile(os.path.join(processed_lst_path, f))]
    files.sort(key=lambda x: datetime.strptime(x[:-3], "%Y%m%d%H%M"))
    
    mask = Dataset(f'{processed_misc_path}/mask.nc')["Band1"][:]
    gaps_pendentes = []

    for file_to_process in files:
        ds_path = f"{processed_lst_path}/{file_to_process}"
        with Dataset(ds_path, 'r+') as ds:
            data = ds["Band1"][:]
            original_data = data.copy()
            gap_mask = np.full_like(data, np.nan)

            for gap in gaps_pendentes:
                x, y, tamanho_gap, duracao = gap
                data[x:x+tamanho_gap, y:y+tamanho_gap] = np.nan
                gap_mask[x:x+tamanho_gap, y:y+tamanho_gap] = 1

            gaps_pendentes = [(x, y, tamanho, duracao-1) for x, y, tamanho, duracao in gaps_pendentes if duracao-1 > 0]

            data, novos_gaps, new_gap = introduzir_gaps(data, mask)

            if new_gap:
                logging.info(f"Gap introduced in: {file_to_process}")
            else:
                logging.info(f"No gap: {file_to_process}")

            gap_mask[np.isnan(original_data)] = np.nan                

            np.savez(f"{processed_lst_path_gaps}/{file_to_process[:-3]}.npz", original_data=original_data, data_with_gaps=data, gap_mask=gap_mask)

            for x, y, tamanho, duracao in novos_gaps:
                if duracao > 1:
                    gaps_pendentes.append((x, y, tamanho, duracao))

#-----------------------------------------------------------------------------------------------------------

# Now, download the files from INMET
#-----------------------------------------------------------------------------------------------------------
def download_inmet():
    os.makedirs(raw_inmet_path, exist_ok=True)
    df_stations = pd.read_csv(stations_csv_path)
    api_token = os.getenv('INMET_API_TOKEN')
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    for index, row in df_stations.iterrows():
        retrieve_data(row['CD_ESTACAO'], start_year, end_year, api_token, raw_inmet_path)

#-----------------------------------------------------------------------------------------------------------

# Unifying inmet data so we don't have to deal with multiple files
#-----------------------------------------------------------------------------------------------------------
def unify_inmet():
    columns = ['CD_ESTACAO', 'DT_MEDICAO', 'HR_MEDICAO', 'TEM_MIN', 'TEM_MAX', 'TEM_INS', 'VL_LATITUDE', 'VL_LONGITUDE']
    dfs = []
    os.makedirs(processed_inmet_path, exist_ok=True)

    nc_files = os.listdir(raw_lst_path)
    lst_reference_file = os.path.join(raw_lst_path, nc_files[0])

    lons, lats = reproject_lats_lons(lst_reference_file)    
    
    for filename in os.listdir(raw_inmet_path):
        logging.info(f"Adding: {filename} to dataset")
        if filename.endswith('.parquet'):
            file_path = os.path.join(raw_inmet_path, filename)
            df = pd.read_parquet(file_path)           
            lat = df['VL_LATITUDE'][0]
            lon = df['VL_LONGITUDE'][0]
            x, y = find_closest_lat_lon(lat, lon, lats, lons)           
            df = df[columns]
            df['x'] = x
            df['y'] = y
            dfs.append(df)
   
    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat.to_parquet(processed_inmet_path+'/concat.parquet')

#-----------------------------------------------------------------------------------------------------------

# Generating the data for inmet WS in our format
#-----------------------------------------------------------------------------------------------------------
def generate_inmet_data():
    df = pd.read_parquet(processed_inmet_path+'/concat.parquet')
    files = [f for f in os.listdir(processed_lst_path_gaps) if os.path.isfile(os.path.join(processed_lst_path_gaps, f))]
    files.sort(key=lambda x: datetime.strptime(x[:-4], "%Y%m%d%H%M"))

    raw_files = [f for f in os.listdir(raw_lst_path) if os.path.isfile(os.path.join(raw_lst_path, f))]
    helper_file_path = os.path.join(raw_lst_path, raw_files[0])
    
    lst_variables = gdal.Open(f'NETCDF:{helper_file_path}:LST')
    metadata = lst_variables.GetMetadata()
    undef = float(metadata.get('LST#_FillValue'))

    for file_to_process in files:
        utc_date = datetime.strptime(file_to_process[:-4], "%Y%m%d%H%M")
        brt_date = utc_date + timedelta(hours=3)

        data = np.full((1086, 1086), np.nan)
        df_filtered = df[(df["DT_MEDICAO"] == brt_date.strftime("%Y-%m-%d")) & (df["HR_MEDICAO"] == brt_date.strftime("%H%M"))]

        if not df_filtered.empty:
            logging.info(f"Creating weather stations data for: {brt_date.strftime('%Y-%m-%d')} {brt_date.strftime('%H%M')}")
            for index, row in df_filtered.iterrows():
                data[row["x"],row["y"]] = row["TEM_INS"]

            reprojected_filename = f'{processed_inmet_path}/{file_to_process[:-4]}.nc'
            reproject(reprojected_filename, lst_variables, data, extent, undef)                
            temp_data = Dataset(reprojected_filename)["Band1"][:]

            indices = np.indices(temp_data.shape)
            non_nan_indices = np.nonzero(~np.isnan(temp_data))
            non_nan_values = temp_data[non_nan_indices]
            interp_func = interpolate.NearestNDInterpolator(non_nan_indices, non_nan_values)
            interpolated_data = interp_func(indices[0], indices[1])            

            saved_data = np.load(os.path.join(processed_lst_path_gaps, file_to_process))
            np.savez(os.path.join(processed_lst_path_gaps, file_to_process), 
                original_data=saved_data["original_data"], 
                data_with_gaps=saved_data["data_with_gaps"], 
                gap_mask=saved_data["gap_mask"], 
                ws_data=interpolated_data)
            saved_data.close()
            #os.remove(reprojected_filename)
            

#-----------------------------------------------------------------------------------------------------------

# Filling the gaps with the inmet data
#-----------------------------------------------------------------------------------------------------------
def fill_gaps():
    files = [f for f in os.listdir(processed_lst_path_gaps) if os.path.isfile(os.path.join(processed_lst_path_gaps, f))]
    files.sort(key=lambda x: datetime.strptime(x[:-4], "%Y%m%d%H%M"))

    for file_to_process in files:
        file_path = os.path.join(processed_lst_path_gaps, file_to_process)
        data = np.load(file_path)
        original_data = data['original_data']
        data_with_gaps = data['data_with_gaps']
        gap_mask = data['gap_mask']
        weather_station_data = data['ws_data']

        filled_data = np.full_like(gap_mask, np.nan)

        non_nan_indices = ~np.isnan(gap_mask)

        filled_data[non_nan_indices] = weather_station_data[non_nan_indices]
        logging.info(f"Filling data for: {file_to_process}")
        # Save the new array back to disk with updated data
        np.savez(file_path, original_data=original_data, data_with_gaps=data_with_gaps, 
                gap_mask=gap_mask, ws_data=weather_station_data, filled_data=filled_data)

#-----------------------------------------------------------------------------------------------------------

# Let's evaluate the result
#-----------------------------------------------------------------------------------------------------------
def evaluate():
    def mean_absolute_error(original, filled):
        """Calculate mean absolute error where filled data is not NaN."""
        mask = ~np.isnan(filled)
        error = np.abs(original[mask] - filled[mask])
        return np.mean(error)

    def root_mean_square_error(original, filled):
        """Calculate root mean square error where filled data is not NaN."""
        mask = ~np.isnan(filled)
        error = (original[mask] - filled[mask]) ** 2
        return np.sqrt(np.mean(error))

    files = [f for f in os.listdir(processed_lst_path_gaps) if os.path.isfile(os.path.join(processed_lst_path_gaps, f))]
    files.sort(key=lambda x: datetime.strptime(x[:-4], "%Y%m%d%H%M"))

    mae_values = []
    rmse_values = []

    for file_name in files:
        file_path = os.path.join(processed_lst_path_gaps, file_name)
        try:
            data = np.load(file_path)
            original_data = data['original_data']
            filled_data = data['filled_data']
            gap_mask = data['gap_mask']

            if np.all(np.isnan(gap_mask)):
                logging.info(f"Skipping file due to all NaN gap_mask: {file_name}")
                continue

            mae = mean_absolute_error(original_data, filled_data)
            rmse = root_mean_square_error(original_data, filled_data)

            mae_values.append(mae)
            rmse_values.append(rmse)

            logging.info(f"Calculated MAE: {mae} and RMSE: {rmse} for file: {file_name}")

        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")

    if mae_values and rmse_values:
        average_mae = np.mean(mae_values)
        average_rmse = np.mean(rmse_values)
        logging.info(f"Average Mean Absolute Error across all files: {average_mae}")
        logging.info(f"Average Root Mean Square Error across all files: {average_rmse}")

        # Plotting the error distributions
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(mae_values, bins='auto', color='blue', alpha=0.7)
        plt.title('Distribution of MAE')
        plt.xlabel('Mean Absolute Error')
        plt.ylabel('Frequency')
        plt.annotate(f'Average MAE: {average_mae:.2f}', xy=(0.70, 0.90), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='white'))

        plt.subplot(1, 2, 2)
        plt.hist(rmse_values, bins='auto', color='red', alpha=0.7)
        plt.title('Distribution of RMSE')
        plt.xlabel('Root Mean Square Error')
        plt.ylabel('Frequency')
        plt.annotate(f'Average RMSE: {average_rmse:.2f}', xy=(0.70, 0.90), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'))

        plt.tight_layout()
        plt.savefig('error_distributions.png')
        plt.show()
    else:
        logging.info("No valid data to process.")


#-----------------------------------------------------------------------------------------------------------

# Finally, execute everything
#-----------------------------------------------------------------------------------------------------------
def main():
    #download_goes()
    #reproject_files()
    #download_inmet()
    #unify_inmet()
    #generate_gaps()
    #generate_inmet_data()
    #fill_gaps()
    evaluate()

if __name__ == "__main__":
    main()