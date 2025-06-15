import logging
import argparse
import pandas as pd
import os
import requests
import time
from tqdm.auto import tqdm


def get_lat_lng_osm(address, delay=1.0, retries=3):
    """
    Geocodifica um endereço usando o Nominatim (OpenStreetMap).
    
    Args:
        address (str): Endereço completo (ou CEP)
        delay (float): Delay entre as requisições
        retries (int): Número de tentativas

    Returns:
        tuple: (latitude, longitude) ou (None, None)
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {'User-Agent': 'Arboseer-Geocoder (academic research project)'}

    params = {
        'q': address,
        'format': 'json',
        'addressdetails': 1,
        'limit': 1
    }

    for attempt in range(retries):
        try:
            time.sleep(delay)
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return lat, lon
            else:
                return None, None
        except Exception as e:
            logging.warning(f"Tentativa {attempt+1} falhou para endereço {address}: {e}")

    return None, None


def preprocess_cnes(parquet_path, output_path):
    """
    Processa o arquivo CNES Parquet, adicionando latitude e longitude via OSM Nominatim.
    """
    try:
        logging.info(f"Lendo arquivo: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        if 'COD_CEP' not in df.columns:
            raise ValueError("A coluna 'COD_CEP' não existe no arquivo de entrada.")

        tqdm.pandas(desc="Geocodificando")

        # Como o Nominatim aceita endereço textual, podemos usar o CEP como string para tentar geocodificar
        df[['LAT', 'LNG']] = df['COD_CEP'].progress_apply(
            lambda cep: get_lat_lng_osm(f"{cep} Brasil") if pd.notnull(cep) else (None, None)
        ).apply(pd.Series)

        df_final = df[['CNES', 'COD_CEP', 'LAT', 'LNG']]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_parquet(output_path)

        logging.info(f"Arquivo processado e salvo em: {output_path}")
    except Exception as e:
        logging.error(f"Erro durante o processamento: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Append lat/lng to CNES Parquet file using OpenStreetMap Nominatim API")
    parser.add_argument("parquet_path", help="Path to the input Parquet file")
    parser.add_argument("output_path", help="Path to the output Parquet file")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    preprocess_cnes(args.parquet_path, args.output_path)


if __name__ == "__main__":
    main()
