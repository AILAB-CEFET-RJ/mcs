import logging
import argparse
import pandas as pd
import os

def extract_sinan_cases(cnes_id, cod_uf, cod_ibge, input_path, output_path, filled, start_date=None, end_date=None):
    """
    Extract cases from a SINAN parquet file

    Args:
    id: str, CNES number
    input_path: str, Parquet file path
    output_path: str, Destination path
    """
    try:
        df = pd.read_parquet(input_path)
        if cnes_id is not None:
            df = df[df["ID_UNIDADE"] == cnes_id]
        if cod_uf is not None:
            df = df[df["SG_UF"] == cod_uf]
        if cod_ibge is not None:
            df = df[df["ID_MUNICIP"] == cod_ibge]

        grouped_df = df.groupby(["ID_UNIDADE", "DT_NOTIFIC"])
        aggregated_df = grouped_df.size().reset_index(name="CASES")
        if filled:
            # Preenchendo lacunas no dataset
            aggregated_df["DT_NOTIFIC"] = pd.to_datetime(aggregated_df["DT_NOTIFIC"])

            # Criando um DataFrame com todas as combinações de datas possíveis
            min_date = pd.to_datetime(start_date, format='%Y-%m-%d') if start_date else aggregated_df["DT_NOTIFIC"].min()
            max_date = pd.to_datetime(end_date, format='%Y-%m-%d') if end_date else aggregated_df["DT_NOTIFIC"].max()
            all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

            all_combinations = pd.MultiIndex.from_product(
                [aggregated_df["ID_UNIDADE"].unique(), all_dates],
                names=["ID_UNIDADE", "DT_NOTIFIC"]
            )

            all_combinations_df = pd.DataFrame(index=all_combinations).reset_index()

            # Mesclando com o DataFrame original
            merged_df = pd.merge(all_combinations_df, aggregated_df, on=["ID_UNIDADE", "DT_NOTIFIC"], how="left")
            merged_df["CASES"] = merged_df["CASES"].fillna(0).astype(int)

            directory = os.path.dirname(output_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")

            merged_df.to_parquet(output_path, index=False)
        else:
            aggregated_df.to_parquet(output_path, index=False)
            aggregated_df.to_csv('teste.csv')

        logging.info(f"Cases extracted at: {output_path}")
    except Exception as e:
        logging.error(f"Error extracting cases: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract cases from a SINAN parquet file")
    parser.add_argument("input_path", help="Path to the input Parquet file")
    parser.add_argument("output_path", help="Path to the output Parquet file")
    parser.add_argument("--cnes_id", dest="cnes_id", help="CNES number", default=None)
    parser.add_argument("--cod_uf", dest="cod_uf", help="UF code of the cases", default=None)
    parser.add_argument("--cod_ibge", dest="cod_ibge", help="IBGE code of the cases", default=None)
    parser.add_argument("--filled", dest="filled", type=bool, nargs='?', const=True, default=False, help="Set if the output will be filled with dates")
    parser.add_argument("--start_date", dest="start_date", help="Start fill date in yyyymmdd format", default=None)
    parser.add_argument("--end_date", dest="end_date", help="End fill date in yyyymmdd format", default=None)
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    if args.output_path is None:
        args.output_path = args.input_path

    extract_sinan_cases(args.cnes_id, args.cod_uf, args.cod_ibge, args.input_path, args.output_path, args.filled, args.start_date, args.end_date)

if __name__ == "__main__":
    main()