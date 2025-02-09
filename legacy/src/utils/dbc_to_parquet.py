import argparse
from pyreaddbc import dbc2dbf
from dbfread import DBF
import pandas as pd
import os
import logging
from tqdm import tqdm

def dbc_to_parquet(dbc_path, parquet_path):
    """
    Convert a DBC file to a Parquet file.
    
    Args:
    dbc_path: str, path to the input DBC file
    parquet_path: str, path to the output Parquet file

    """
    try:
        if not dbc_path.lower().endswith(".dbc"):
            raise ValueError("The provided file does not have a .dbc extension")        

        directory = os.path.dirname(parquet_path)
        if not os.path.exists(directory):
            os.makedirs(directory)            

        dbf_path = os.path.splitext(dbc_path)[0] + ".dbf"
        
        # Convert to dbf since no straight conversion exists
        dbc2dbf(dbc_path, dbf_path)

        dbf = DBF(dbf_path, encoding="iso-8859-1")
        frame = pd.DataFrame(iter(tqdm(dbf, total=len(dbf), desc="Converting DBF to DataFrame")))

        frame.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

        # Clean up temp DBF file
        os.remove(dbf_path)

        logging.info(f"File converted to parquet at: {parquet_path}")
    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
        raise
    except IOError as e:
        logging.error(f"I/O error: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert DBC to Parquet")
    parser.add_argument("dbc_path", help="Path to the input DBC file")
    parser.add_argument("parquet_path", help="Path to the output Parquet file")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
        
    dbc_to_parquet(args.dbc_path, args.parquet_path)

if __name__ == "__main__":
    main()
