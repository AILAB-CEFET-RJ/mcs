# Download and process CNES Health Units data - OK
python src/ingestion/download_cnes.py ST RJ 2401 data/raw/cnes/STRJ2401.dbc
python src/data/dbc_to_parquet.py data/raw/cnes/STRJ2401.dbc data/raw/cnes/STRJ2401.parquet
python src/preprocess/preprocess_cnes.py data/raw/cnes/STRJ2401.parquet data/processed/cnes/STRJ2401.parquet

# Download and process SINAN dengue cases - OK
python src/ingestion/download_sinan.py DENG 2013 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2014 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2015 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2016 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2017 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2018 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2019 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2020 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2021 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2022 data/raw/sinan
python src/ingestion/download_sinan.py DENG 2023 data/raw/sinan
python src/preprocess/unify_sinan.py data/raw/sinan data/processed/sinan
python src/preprocess/extract_sinan_cases.py data/processed/sinan/concat.parquet data/processed/sinan/DENG.parquet --cod_uf 3304557