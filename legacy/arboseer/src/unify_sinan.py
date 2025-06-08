import logging
import pandas as pd
import argparse
import os
import re

# Required columns for processing
REQUIRED_COLUMNS = {"ID_MUNICIP", "ID_UNIDADE", "DT_NOTIFIC", "SG_UF"}

def extract_year_from_filename(filename):
    """Extracts the year from a filename like 'DENGBR02' -> 2002."""
    match = re.search(r'DENGBR(\d{2})', filename)
    if match:
        year = int(match.group(1))
        return 2000 + year if year >= 0 else None
    return None

def unify_sinan(input_path, output_path):
    dfs = []
    os.makedirs(output_path, exist_ok=True)
    
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    for file in files:
        logging.info(f"Processing: {file}")

        # Extract the expected year from the filename
        expected_year = extract_year_from_filename(file)
        if expected_year is None:
            logging.warning(f"Skipping {file} - Cannot determine year from filename")
            continue

        if file.endswith('.parquet'):
            file_path = os.path.join(input_path, file)
            df = pd.read_parquet(file_path)

            # Ensure required columns exist
            missing_columns = REQUIRED_COLUMNS - set(df.columns)
            if missing_columns:
                logging.warning(f"Skipping {file} - missing columns: {missing_columns}")
                continue  # Skip this file if columns are missing

            # Convert DT_NOTIFIC to datetime and filter by year
            df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
            df = df[df["DT_NOTIFIC"].dt.year == expected_year]

            # Standardize ID_UNIDADE to be a 7-digit string
            df["ID_UNIDADE"] = df["ID_UNIDADE"].astype(str).str.zfill(7)

            # Keep only required fields
            df = df[["ID_MUNICIP", "ID_UNIDADE", "DT_NOTIFIC", "SG_UF"]]

            dfs.append(df)

    if not dfs:
        logging.error("No valid data found. Ensure files contain the required columns and match the expected year.")
        return

    # Concatenate all data
    df_concat = pd.concat(dfs, ignore_index=True)

    # Save to Parquet
    output_file = os.path.join(output_path, 'concat.parquet')  # TODO: Move file name to args
    df_concat.to_parquet(output_file)

    logging.info(f"Saved concatenated SINAN file at: {output_file}")

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
