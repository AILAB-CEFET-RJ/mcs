import logging
import pandas as pd
import argparse
import os

def unify_sinan(input_path, output_path):
    dfs = []
    os.makedirs(output_path, exist_ok=True)
    
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    for file in files:
        logging.info(f"Adding: {file} to dataset")
        if file.endswith('.parquet'):
            file_path = os.path.join(input_path, file)
            df = pd.read_parquet(file_path)           

            dfs.append(df)
   
    df_concat = pd.concat(dfs, ignore_index=True)
    
    df_concat.to_parquet(os.path.join(output_path, 'concat.parquet')) #TODO: Move the file name to args

    logging.info(f"Saved concatenated inmet file at: {output_path}/concat.parquet") #TODO: Move the file name to args

def main():
    parser = argparse.ArgumentParser(description="Unify SINAN datasets")
    parser.add_argument("input_path", help="Input path")
    parser.add_argument("output_path", help="Output path")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    unify_sinan(args.input_path, args.output_path)

if __name__ == "__main__":
    main()