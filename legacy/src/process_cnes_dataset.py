import logging
import argparse
import pandas as pd
import os
import requests
import time
from tqdm.auto import tqdm


def get_lat_lng(cep):      
    """
    Gets latitude and longitude for a givem cep

    Args:
    cep: str, numbers only cep

    """    
    time.sleep(1)
    url = f"https://cep.awesomeapi.com.br/json/{cep}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        lat = data.get('lat', 'Default value if not found')
        lng = data.get('lng', 'Default value if not found')
        return lat, lng
    return None, None

def process_cnes_dataset(parquet_path, output_path):
    """
    Process a CNES ST parquet file

    Args:
    parquet_path: str, Parquet file path
    output_path: str, Destination path

    """        
    try:
        df = pd.read_parquet(parquet_path)

        tqdm.pandas(desc="Processing Rows", total=df.shape[0])
        
        df[['LAT', 'LNG']] = df.progress_apply(lambda row: get_lat_lng(row['COD_CEP']), axis=1, result_type='expand')        
        df = df[['CNES', 'COD_CEP', 'LAT', 'LNG']]

        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

        df.to_parquet(output_path)

        logging.info(f"Dataset processed at: {output_path}")    
    except Exception as e:
        logging.error(f"Erro processing dataset: {e}")    
 
def main():
    parser = argparse.ArgumentParser(description="Append lat/long to CNES Estabelecimentos parquet file")
    parser.add_argument("parquet_path", help="Path to the input Parquet file")
    parser.add_argument("output_path", help="Path to the output Parquet file")    
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")

    args = parser.parse_args()    

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")   
    
    if args.output_path is None:
        args.output_path = args.parquet_path
    
    process_cnes_dataset(args.parquet_path, args.output_path)  

if __name__ == "__main__":
    main()