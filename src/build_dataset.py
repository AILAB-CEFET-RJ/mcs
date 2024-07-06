import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import argparse
import logging

def find_nearest(lat, lon, tree, coords):
    dist, idx = tree.query([[lat, lon]], k=1)
    return coords[idx[0]]

def build_dataset(sinan_path, cnes_path, inmet_path, lst_path, rrqpe_path, output_path, start_date=None, end_date=None):
    # Configurar logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Carregar os datasets dos arquivos Parquet
    logging.info("Carregando os datasets...")
    sinan_df = pd.read_parquet(sinan_path)
    cnes_df = pd.read_parquet(cnes_path)
    inmet_df = pd.read_parquet(inmet_path)
    lst_df = pd.read_parquet(lst_path)
    rrqpe_df = pd.read_parquet(rrqpe_path)
    
    logging.debug(f"sinan_df shape: {sinan_df.shape}")
    logging.debug(f"cnes_df shape: {cnes_df.shape}")
    logging.debug(f"inmet_df shape: {inmet_df.shape}")
    logging.debug(f"lst_df shape: {lst_df.shape}")
    logging.debug(f"rrqpe_df shape: {rrqpe_df.shape}")

    # Converter campos de data para datetime
    logging.info("Convertendo campos de data para datetime...")
    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'], format='%Y%m%d')
    inmet_df['DT_MEDICAO'] = pd.to_datetime(inmet_df['DT_MEDICAO'], format='%Y-%m-%d')
    lst_df['date'] = pd.to_datetime(lst_df['date'], format='%Y%m%d')
    rrqpe_df['date'] = pd.to_datetime(rrqpe_df['date'], format='%Y%m%d')

    # Converter ID_UNIDADE para string
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].astype(str)
    cnes_df['CNES'] = cnes_df['CNES'].astype(str)

    # Filtrar sinan_df para considerar apenas ID_UNIDADE '2296306'
    #logging.info("Filtrando sinan_df para ID_UNIDADE '2296306'...")
    #sinan_df = sinan_df[sinan_df['ID_UNIDADE'] == '2296306']
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

    # Renomear colunas
    logging.info("Renomeando colunas...")
    inmet_df.rename(columns={
        'TEM_MIN': 'TEM_MIN_INMET',
        'TEM_MAX': 'TEM_MAX_INMET',
        'TEM_AVG': 'TEM_AVG_INMET',
        'CHUVA': 'CHUVA_INMET'
    }, inplace=True)

    lst_df.rename(columns={
        'LST_AVG': 'TEM_AVG_SAT',
        'LST_MIN': 'TEM_MIN_SAT',
        'LST_MAX': 'TEM_MAX_SAT'
    }, inplace=True)

    rrqpe_df.rename(columns={
        'RRQPE_SUM': 'CHUVA_SAT'
    }, inplace=True)

    # Criar árvores k-d para busca rápida
    logging.info("Criando árvores k-d para busca rápida...")
    inmet_coords = inmet_df[['VL_LATITUDE', 'VL_LONGITUDE']].values
    lst_coords = lst_df[['latitude', 'longitude']].values
    rrqpe_coords = rrqpe_df[['latitude', 'longitude']].values

    tree_inmet = cKDTree(inmet_coords)
    tree_lst = cKDTree(lst_coords)
    tree_rrqpe = cKDTree(rrqpe_coords)

    # Encontrar a estação meteorológica mais próxima de cada unidade de saúde
    logging.info("Encontrando a estação meteorológica mais próxima...")
    nearest_inmet = np.apply_along_axis(lambda x: find_nearest(x[0], x[1], tree_inmet, inmet_coords), 1, sinan_df[['LAT', 'LNG']].values)
    nearest_lst = np.apply_along_axis(lambda x: find_nearest(x[0], x[1], tree_lst, lst_coords), 1, sinan_df[['LAT', 'LNG']].values)
    nearest_rrqpe = np.apply_along_axis(lambda x: find_nearest(x[0], x[1], tree_rrqpe, rrqpe_coords), 1, sinan_df[['LAT', 'LNG']].values)

    # Adicionar as coordenadas mais próximas aos dados
    sinan_df['closest_LAT_INMET'] = nearest_inmet[:, 0]
    sinan_df['closest_LNG_INMET'] = nearest_inmet[:, 1]
    sinan_df['closest_LAT_SAT'] = nearest_lst[:, 0]
    sinan_df['closest_LNG_SAT'] = nearest_lst[:, 1]
    sinan_df['closest_LAT_RAIN_SAT'] = nearest_rrqpe[:, 0]
    sinan_df['closest_LNG_RAIN_SAT'] = nearest_rrqpe[:, 1]

    # Mesclar com dados do INMET
    logging.info("Mesclando sinan_df com inmet_df...")
    sinan_df = pd.merge(sinan_df, inmet_df, left_on=['closest_LAT_INMET', 'closest_LNG_INMET', 'DT_NOTIFIC'], right_on=['VL_LATITUDE', 'VL_LONGITUDE', 'DT_MEDICAO'], how='left')
    logging.debug(f"sinan_df shape after merging with inmet_df: {sinan_df.shape}")

    # Mesclar com dados do LST
    logging.info("Mesclando sinan_df com lst_df...")
    sinan_df = pd.merge(sinan_df, lst_df, left_on=['closest_LAT_SAT', 'closest_LNG_SAT', 'DT_NOTIFIC'], right_on=['latitude', 'longitude', 'date'], how='left')
    logging.debug(f"sinan_df shape after merging with lst_df: {sinan_df.shape}")

    # Mesclar com dados do RRQPE
    logging.info("Mesclando sinan_df com rrqpe_df...")
    sinan_df = pd.merge(sinan_df, rrqpe_df, left_on=['closest_LAT_RAIN_SAT', 'closest_LNG_RAIN_SAT', 'DT_NOTIFIC'], right_on=['latitude', 'longitude', 'date'], how='left')
    logging.debug(f"sinan_df shape after merging with rrqpe_df: {sinan_df.shape}")

    # Criar features de temperatura e precipitação
    logging.info("Criando features...")

    # Temperatura ideal e extrema
    sinan_df['IDEAL_TEMP_INMET'] = sinan_df['TEM_AVG_INMET'].apply(lambda x: 1 if 21 <= x <= 27 else 0)
    sinan_df['EXTREME_TEMP_INMET'] = sinan_df['TEM_AVG_INMET'].apply(lambda x: 1 if x <= 14 or x >= 38 else 0)
    sinan_df['IDEAL_TEMP_SAT'] = sinan_df['TEM_AVG_SAT'].apply(lambda x: 1 if 21 <= x <= 27 else 0)
    sinan_df['EXTREME_TEMP_SAT'] = sinan_df['TEM_AVG_SAT'].apply(lambda x: 1 if x <= 14 or x >= 38 else 0)

    # Precipitação significativa e extrema
    sinan_df['SIGNIFICANT_RAIN_INMET'] = sinan_df['CHUVA_INMET'].apply(lambda x: 1 if 10 <= x < 150 else 0)
    sinan_df['EXTREME_RAIN_INMET'] = sinan_df['CHUVA_INMET'].apply(lambda x: 1 if x >= 150 else 0)
    sinan_df['SIGNIFICANT_RAIN_SAT'] = sinan_df['CHUVA_SAT'].apply(lambda x: 1 if 10 <= x < 150 else 0)
    sinan_df['EXTREME_RAIN_SAT'] = sinan_df['CHUVA_SAT'].apply(lambda x: 1 if x >= 150 else 0)

    # Amplitude térmica
    sinan_df['TEMP_RANGE_INMET'] = sinan_df['TEM_MAX_INMET'] - sinan_df['TEM_MIN_INMET']
    sinan_df['TEMP_RANGE_SAT'] = sinan_df['TEM_MAX_SAT'] - sinan_df['TEM_MIN_SAT']

    # Médias móveis e acumulados de temperatura e precipitação
    windows = [7, 14, 21]
    for window in windows:
        sinan_df[f'TEM_AVG_INMET_MM_{window}'] = sinan_df['TEM_AVG_INMET'].rolling(window=window).mean()
        sinan_df[f'CHUVA_INMET_MM_{window}'] = sinan_df['CHUVA_INMET'].rolling(window=window).mean()
        sinan_df[f'TEMP_RANGE_INMET_MM_{window}'] = sinan_df['TEMP_RANGE_INMET'].rolling(window=window).mean()
        sinan_df[f'TEM_AVG_SAT_MM_{window}'] = sinan_df['TEM_AVG_SAT'].rolling(window=window).mean()
        sinan_df[f'CHUVA_SAT_MM_{window}'] = sinan_df['CHUVA_SAT'].rolling(window=window).mean()
        sinan_df[f'TEMP_RANGE_SAT_MM_{window}'] = sinan_df['TEMP_RANGE_SAT'].rolling(window=window).mean()
        sinan_df[f'TEM_AVG_INMET_ACC_{window}'] = sinan_df['TEM_AVG_INMET'].rolling(window=window).sum()
        sinan_df[f'CHUVA_INMET_ACC_{window}'] = sinan_df['CHUVA_INMET'].rolling(window=window).sum()
        sinan_df[f'TEM_AVG_SAT_ACC_{window}'] = sinan_df['TEM_AVG_SAT'].rolling(window=window).sum()
        sinan_df[f'CHUVA_SAT_ACC_{window}'] = sinan_df['CHUVA_SAT'].rolling(window=window).sum()

    # Criar features de casos de dengue
    sinan_df['CASES_MM_14'] = sinan_df['CASES'].rolling(window=14).mean()
    sinan_df['CASES_MM_21'] = sinan_df['CASES'].rolling(window=21).mean()
    sinan_df['CASES_ACC_14'] = sinan_df['CASES'].rolling(window=14).sum()
    sinan_df['CASES_ACC_21'] = sinan_df['CASES'].rolling(window=21).sum()

    # Selecionar as colunas necessárias
    logging.info("Selecionando as colunas necessárias...")
    selected_columns = [
        'ID_UNIDADE', 'DT_NOTIFIC', 'LAT', 'LNG', 'closest_LAT_INMET', 'closest_LNG_INMET', 'closest_LAT_SAT', 'closest_LNG_SAT', 'closest_LAT_RAIN_SAT', 'closest_LNG_RAIN_SAT',
        'TEM_MIN_INMET', 'TEM_MAX_INMET', 'TEM_AVG_INMET', 'CHUVA_INMET',
        'TEM_AVG_SAT', 'TEM_MIN_SAT', 'TEM_MAX_SAT', 'CHUVA_SAT',
        'IDEAL_TEMP_INMET', 'EXTREME_TEMP_INMET', 'SIGNIFICANT_RAIN_INMET', 'EXTREME_RAIN_INMET',
        'IDEAL_TEMP_SAT', 'EXTREME_TEMP_SAT', 'SIGNIFICANT_RAIN_SAT', 'EXTREME_RAIN_SAT',
        'TEMP_RANGE_INMET', 'TEMP_RANGE_SAT',
        'CASES', 'CASES_MM_14', 'CASES_MM_21', 'CASES_ACC_14', 'CASES_ACC_21'
    ] + [
        f'TEM_AVG_INMET_MM_{window}' for window in windows
    ] + [
        f'CHUVA_INMET_MM_{window}' for window in windows
    ] + [
        f'TEMP_RANGE_INMET_MM_{window}' for window in windows
    ] + [
        f'TEM_AVG_SAT_MM_{window}' for window in windows
    ] + [
        f'CHUVA_SAT_MM_{window}' for window in windows
    ] + [
        f'TEMP_RANGE_SAT_MM_{window}' for window in windows
    ] + [
        f'TEM_AVG_INMET_ACC_{window}' for window in windows
    ] + [
        f'CHUVA_INMET_ACC_{window}' for window in windows
    ] + [
        f'TEM_AVG_SAT_ACC_{window}' for window in windows
    ] + [
        f'CHUVA_SAT_ACC_{window}' for window in windows
    ]

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
    logging.info(f"Dataset com features salvo em {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Build dataset with features")
    parser.add_argument("sinan_path", help="Path to SINAN data")
    parser.add_argument("cnes_path", help="Path to CNES data")
    parser.add_argument("inmet_path", help="Path to INMET data")
    parser.add_argument("lst_path", help="Path to temperature data")
    parser.add.argument("rrqpe_path", help="Path to rainfall data")
    parser.add.argument("output_path", help="Output path")
    parser.add.argument("--start_date", help="Start date for filtering (YYYY-MM-DD)", default=None)
    parser.add.argument("--end_date", help="End date for filtering (YYYY-MM-DD)", default=None)
    parser.add.argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")

    args = parser.parse_args()

    build_dataset(
        sinan_path=args.sinan_path,
        cnes_path=args.cnes_path,
        inmet_path=args.inmet_path,
        lst_path=args.lst_path,
        rrqpe_path=args.rrqpe_path,
        output_path=args.output_path,
        start_date=args.start_date,
        end_date=args.end_date
    )

if __name__ == "__main__":
    #main()
    build_dataset(
        sinan_path="data/processed/sinan/DENG.parquet",
        cnes_path="data/processed/cnes/STRJ2401.parquet",
        inmet_path="data/processed/inmet/aggregated.parquet",
        lst_path="data/processed/lst/lst.parquet",
        rrqpe_path="data/processed/rrqpe/rrqpe.parquet",
        output_path="data/processed/sinan/sinan.parquet",
        start_date="2020-01-01",
        end_date="2023-12-31"
    )