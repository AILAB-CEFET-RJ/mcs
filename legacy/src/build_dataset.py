import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import argparse
import logging
import xarray as xr

def find_nearest(lat, lon, tree, coords):
    dist, idx = tree.query([[lat, lon]], k=1)
    return coords[idx[0]]

def extract_era5_data(nc_file, lat, lon, date):
    """Extract ERA5 skin temperature data for a specific location and date."""
    try:
        ds = xr.open_dataset(nc_file)
    except FileNotFoundError:
        raise ValueError(f"File not found: {nc_file}")
    
    # Select the nearest point for latitude and longitude
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')

    # Select data for the specific date (all times in the date)
    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
    day_data = ds_point.sel(time=slice(f"{date_str}T00:00:00", f"{date_str}T23:00:00"))

    if day_data.time.size == 0:
        raise ValueError(f"No data available for {date_str} at lat={lat}, lon={lon}")

    # Calculate daily mean, min, and max for skin temperature (skt)
    tem_avg = day_data['skt'].mean().item() - 273.15
    tem_min = day_data['skt'].min().item() - 273.15
    tem_max = day_data['skt'].max().item() - 273.15

    return tem_avg, tem_min, tem_max

def build_dataset(sinan_path, cnes_path, era5_path, output_path, start_date=None, end_date=None):
    # Configurar logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Carregar os datasets dos arquivos Parquet
    logging.info("Carregando os datasets...")
    sinan_df = pd.read_parquet(sinan_path)
    cnes_df = pd.read_parquet(cnes_path)

    # Converter campos de data para datetime
    logging.info("Convertendo campos de data para datetime...")
    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'], format='%Y%m%d')

    # Converter ID_UNIDADE para string
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].astype(str)
    cnes_df['CNES'] = cnes_df['CNES'].astype(str)

    # Filtrar sinan_df para considerar apenas ID_UNIDADE '2296306'
    #logging.info("Filtrando sinan_df para ID_UNIDADE '2296306'...")
    sinan_df = sinan_df[sinan_df['ID_UNIDADE'].isin([
                        '7427549',
                        '2268922',
                        '7149328',
                        '2299216',
                        '0106453',
                        '6870066',
                        '6042619',
                        '2288893',
                        '5106702',
                        '6635148',
                        '2269481',
                        '2708353',
                        '7591136',
                        '2283395',
                        '2287579',
                        '2291533',
                        '2292386',
                        '0012505',
                        '2292084',
                        '6518893'])]
    #logging.debug(f"sinan_df shape after filtering: {sinan_df.shape}")

    # Processar sinan_df
    logging.info("Processando sinan_df...")
    sinan_df.dropna(subset=['ID_UNIDADE'], inplace=True)
    logging.debug(f"sinan_df shape after processing: {sinan_df.shape}")

    # Processar cnes_df
    logging.info("Processando cnes_df...")
    cnes_df = cnes_df[cnes_df['CNES'].isin(sinan_df['ID_UNIDADE'])]
    cnes_df.rename(columns={'CNES': 'ID_UNIDADE'}, inplace=True)
    cnes_df['LAT'] = pd.to_numeric(cnes_df['LAT'], errors='coerce')
    cnes_df['LNG'] = pd.to_numeric(cnes_df['LNG'], errors='coerce')
    cnes_df.dropna(subset=['LAT', 'LNG'], inplace=True)
    logging.debug(f"cnes_df shape after processing: {cnes_df.shape}")

    # Mesclar SINAN com CNES para adicionar lat/lng
    logging.info("Mesclando sinan_df com cnes_df...")
    sinan_df = pd.merge(sinan_df, cnes_df[['ID_UNIDADE', 'LAT', 'LNG']], on='ID_UNIDADE', how='left')
    sinan_df.dropna(subset=['LAT', 'LNG'], inplace=True)
    logging.debug(f"sinan_df shape after merging with cnes_df: {sinan_df.shape}")

    # Criando campos do ERA5
    sinan_df['TEM_AVG_ERA5'] = np.nan
    sinan_df['TEM_MIN_ERA5'] = np.nan
    sinan_df['TEM_MAX_ERA5'] = np.nan

    logging.info("Extraindo dados do ERA5 e adicionando ao dataset...")
    era5_dates = sinan_df['DT_NOTIFIC'].unique()
    era5_coords = sinan_df[['LAT', 'LNG']].drop_duplicates()    
    era5_tree = cKDTree(era5_coords[['LAT', 'LNG']].values)
    for date in era5_dates:
        for lat, lon in era5_coords.values:
            nearest_coords = find_nearest(lat, lon, era5_tree, era5_coords.values)
            era5_avg, era5_min, era5_max = extract_era5_data(era5_path, nearest_coords[0], nearest_coords[1], date)

            # Update the corresponding rows in SINAN dataset
            mask = (sinan_df['DT_NOTIFIC'] == date) & (sinan_df['LAT'] == nearest_coords[0]) & (sinan_df['LNG'] == nearest_coords[1])
            sinan_df.loc[mask, 'TEM_AVG_ERA5'] = era5_avg
            sinan_df.loc[mask, 'TEM_MIN_ERA5'] = era5_min
            sinan_df.loc[mask, 'TEM_MAX_ERA5'] = era5_max          

    # Criar features de temperatura e precipitação
    logging.info("Criando features...")

    # Temperatura ideal e extrema
    sinan_df['IDEAL_TEMP_ERA5'] = sinan_df['TEM_AVG_ERA5'].apply(lambda x: 1 if 21 <= x <= 27 else 0)
    sinan_df['EXTREME_TEMP_ERA5'] = sinan_df['TEM_AVG_ERA5'].apply(lambda x: 1 if x <= 14 or x >= 38 else 0)    

    # Amplitude térmica
    sinan_df['TEMP_RANGE_ERA5'] = sinan_df['TEM_MAX_ERA5'] - sinan_df['TEM_MIN_ERA5']    

    # Médias móveis e acumulados de temperatura e precipitação
    windows = [7, 14, 21]
    for window in windows:
        # Média Móvel (MM) da temperatura
        sinan_df[f'TEM_AVG_ERA5_MM_{window}'] = sinan_df['TEM_AVG_ERA5'].rolling(window=window).mean()

        # Média Móvel (MM) da amplitude térmica
        sinan_df[f'TEMP_RANGE_ERA5_MM_{window}'] = sinan_df['TEMP_RANGE_ERA5'].rolling(window=window).mean()

        # Temperatura acumulada (ACC) na janela
        sinan_df[f'TEM_AVG_ERA5_ACC_{window}'] = sinan_df['TEM_AVG_ERA5'].rolling(window=window).sum()        

    # Criar features de casos de dengue
    sinan_df['CASES_MM_14'] = sinan_df['CASES'].rolling(window=14).mean()
    sinan_df['CASES_MM_21'] = sinan_df['CASES'].rolling(window=21).mean()
    sinan_df['CASES_ACC_14'] = sinan_df['CASES'].rolling(window=14).sum()
    sinan_df['CASES_ACC_21'] = sinan_df['CASES'].rolling(window=21).sum()

    # Selecionar as colunas necessárias
    logging.info("Selecionando as colunas necessárias...")
    base_columns = ['ID_UNIDADE', 'DT_NOTIFIC', 'LAT', 'LNG']
    
    cases_coluns = ['CASES', 'CASES_MM_14', 'CASES_MM_21', 'CASES_ACC_14', 'CASES_ACC_21']

    era5_columns = ['TEM_MIN_ERA5', 'TEM_AVG_ERA5', 'TEM_MAX_ERA5',
                   'IDEAL_TEMP_ERA5', 'EXTREME_TEMP_ERA5', 'TEMP_RANGE_ERA5'] + [
                    f'TEM_AVG_ERA5_MM_{window}' for window in windows ] + [
                    f'TEMP_RANGE_ERA5_MM_{window}' for window in windows ] + [
                    f'TEM_AVG_ERA5_ACC_{window}' for window in windows ]

    selected_columns = base_columns + cases_coluns + era5_columns

    final_df = sinan_df[selected_columns]
    logging.debug(f"final_df shape after selecting columns: {final_df.shape}")

    # Filtrar por data de início e fim, se fornecidas
    if start_date:
        logging.info(f"Filtrando por data de início: {start_date}")
        final_df = final_df[final_df['DT_NOTIFIC'] >= pd.to_datetime(start_date)]
    if end_date:
        logging.info(f"Filtrando por data de fim: {end_date}")
        final_df = final_df[final_df['DT_NOTIFIC'] <= pd.to_datetime(end_date)]
    logging.debug(f"final_df shape after date filtering: {final_df.shape}")

    # Salvar o dataset final como um arquivo Parquet
    logging.info(f"Salvando o dataset final em {output_path}")
    final_df.to_parquet(output_path, index=False)
    final_df.to_csv('final.csv')
    logging.info(f"Dataset com features salvo em {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Build dataset with features")
    parser.add_argument("sinan_path", help="Path to SINAN data")
    parser.add_argument("cnes_path", help="Path to CNES data")
    parser.add_argument("era5_path", help="Path to ERA5 data")
    parser.add_argument("output_path", help="Output path")
    parser.add_argument("--start_date", help="Start date for filtering (YYYY-MM-DD)", default=None)
    parser.add_argument("--end_date", help="End date for filtering (YYYY-MM-DD)", default=None)
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")

    args = parser.parse_args()

    build_dataset(
        sinan_path=args.sinan_path,
        cnes_path=args.cnes_path,
        era5_path=args.era5_path,
        output_path=args.output_path,
        start_date=args.start_date,
        end_date=args.end_date
    )

if __name__ == "__main__":
    main()
    #build_dataset(
    #     sinan_path="data/processed/sinan/DENG.parquet",
    #     cnes_path="data/processed/cnes/STRJ2401.parquet",
    #     era5_path="data/raw/era5/RJ_1997_2024.nc",
    #     output_path="data/processed/sinan/sinan.parquet",
    #     start_date="2020-01-01",
    #     end_date="2023-12-31"
    #)