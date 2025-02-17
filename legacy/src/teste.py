import pandas as pd
import argparse
import logging

def top_id_unidade(input_file):
    """
    Reads a Parquet file and finds the top 10 ID_UNIDADE with the most CASES.
    
    Args:
        input_file (str): Path to the Parquet file.
    """
    logging.info(f"Loading data from {input_file}")
    
    # Load the dataset
    df = pd.read_parquet(input_file)

    # Ensure CASES column exists
    if "CASES" not in df.columns:
        logging.error("Column 'CASES' not found in the dataset.")
        return

    # Ensure ID_UNIDADE exists
    if "ID_UNIDADE" not in df.columns:
        logging.error("Column 'ID_UNIDADE' not found in the dataset.")
        return

    # Group by ID_UNIDADE and sum CASES
    top_units = df.groupby("ID_UNIDADE")["CASES"].sum().nlargest(10)

    print("\nTop 10 ID_UNIDADE with the most CASES:")
    print(top_units)

def main():
    parser = argparse.ArgumentParser(description="Find top 10 ID_UNIDADE with the most CASES.")
    parser.add_argument("input_file", help="Path to the input Parquet file")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    top_id_unidade(args.input_file)

if __name__ == "__main__":
    #main()
    top_id_unidade("data/processed/sinan/DENG.parquet")
