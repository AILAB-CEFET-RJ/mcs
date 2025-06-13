import os
import argparse
import logging
import yaml
from tqdm import tqdm

# Importação dos módulos existentes
from download.download_cnes import download_cnes
from download.download_sinan import download_sinan, combine_parquet_files
from preprocess.unify_sinan import unify_sinan
from preprocess.extract_sinan_cases import extract_sinan_cases
from preprocess.process_cnes import preprocess_cnes

# Definir paths
BASE_DIR = os.path.abspath(".")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Carregar configuração YAML
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def download_cnes_step(config):
    logging.info("🔽 Download CNES...")

    os.makedirs(os.path.join(RAW_DIR, "cnes"), exist_ok=True)
    dbc_path = os.path.join(RAW_DIR, "cnes", f"S{config['cnes']['state_code']}{config['cnes']['file_code']}.dbc")

    download_cnes(config['cnes']['tipo'], config['cnes']['state_code'], config['cnes']['file_code'], dbc_path)

def process_cnes_step(config):
    logging.info("🔄 Processamento CNES...")
    os.makedirs(os.path.join(PROCESSED_DIR, "cnes"), exist_ok=True)

    dbc_path = os.path.join(RAW_DIR, "cnes", f"S{config['cnes']['state_code']}{config['cnes']['file_code']}.dbc")
    parquet_path = os.path.join(RAW_DIR, "cnes", f"S{config['cnes']['state_code']}{config['cnes']['file_code']}.parquet")
    processed_path = os.path.join(PROCESSED_DIR, "cnes", f"S{config['cnes']['state_code']}{config['cnes']['file_code']}.parquet")

    os.system(f"python src/data/dbc_to_parquet.py {dbc_path} {parquet_path}")
    preprocess_cnes(parquet_path, processed_path)

def download_sinan_step(config):
    logging.info("🔽 Download SINAN...")
    os.makedirs(os.path.join(RAW_DIR, "sinan"), exist_ok=True)

    for year in tqdm(config["sinan"]["years"], desc="SINAN"):
        downloaded_file = download_sinan(
            config["sinan"]["disease"], year, os.path.join(RAW_DIR, "sinan")
        )
        if downloaded_file:
            file_path = f"{os.path.join(RAW_DIR, 'sinan')}/{downloaded_file.name}.parquet"
            combine_parquet_files(file_path, file_path)

def process_sinan_step(config):
    logging.info("🔬 Processamento SINAN...")
    os.makedirs(os.path.join(PROCESSED_DIR, "sinan"), exist_ok=True)

    unify_sinan(os.path.join(RAW_DIR, "sinan"), os.path.join(PROCESSED_DIR, "sinan"))
    
    cnes_id = None 
    if str(config["sinan"]["cnes_id"]) != "None":
        cnes_id = str(config["sinan"]["cnes_id"]) 
        
    cod_uf = None 
    if str(config["sinan"]["cod_uf"]) != "None":
        cod_uf = str(config["sinan"]["cod_uf"]) 
        
    cod_ibge = None 
    if str(config["sinan"]["cod_ibge"]) != "None":
        cod_ibge = str(config["sinan"]["cod_ibge"])                 
    
    extract_sinan_cases(
        cnes_id=cnes_id,
        cod_uf=cod_uf,
        cod_ibge=cod_ibge,
        input_path=os.path.join(PROCESSED_DIR, "sinan", "concat.parquet"),
        output_path=os.path.join(PROCESSED_DIR, "sinan", f"{config['sinan']['disease']}.parquet"),
        filled=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arboseer Ingestion Pipeline")

    parser.add_argument("--config", type=str, default="config/ingestion.yaml", help="Arquivo YAML de configuração")
    parser.add_argument("--download-cnes", action="store_true", help="Download CNES")
    parser.add_argument("--process-cnes", action="store_true", help="Processamento CNES")
    parser.add_argument("--download-sinan", action="store_true", help="Download SINAN")
    parser.add_argument("--process-sinan", action="store_true", help="Processamento SINAN")

    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log), format="%(asctime)s - %(levelname)s - %(message)s")
    config = load_config(args.config)

    if args.download_cnes:
        download_cnes_step(config)

    if args.process_cnes:
        process_cnes_step(config)

    if args.download_sinan:
        download_sinan_step(config)

    if args.process_sinan:
        process_sinan_step(config)
