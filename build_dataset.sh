# Download and process CNES Health Units data - OK
#python src/utils/download_cnes_file.py ST RJ 2401 data/raw/cnes/STRJ2401.dbc
#python src/utils/dbc_to_parquet.py data/raw/cnes/STRJ2401.dbc data/raw/cnes/STRJ2401.parquet
#python src/process_cnes_dataset.py data/raw/cnes/STRJ2401.parquet data/processed/cnes/STRJ2401.parquet

# we won't be using 2019, it's just being downloaded to have a more complete 2020

# Download and process SINAN dengue cases - OK
#python src/utils/download_sinan_file.py DENG 2019 data/raw/sinan
#python src/utils/download_sinan_file.py DENG 2020 data/raw/sinan
#python src/utils/download_sinan_file.py DENG 2021 data/raw/sinan
#python src/utils/download_sinan_file.py DENG 2022 data/raw/sinan
#python src/utils/download_sinan_file.py DENG 2023 data/raw/sinan
#python src/unify_sinan.py data/raw/sinan data/processed/sinan
#python src/extract_sinan_cases.py data/processed/sinan/concat.parquet data/processed/sinan/DENG.parquet --cod_uf 33 --filled --start_date 2019-01-01 --end_date 2023-12-31

# Download inmet data - OK
#python src/utils/download_inmet_data.py -s A617 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A615 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A628 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A606 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A502 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A604 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A607 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A620 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A629 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A557 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A603 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A518 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A608 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A517 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A627 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A624 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A570 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A513 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A619 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A529 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A610 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A622 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A637 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A609 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A626 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A652 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A636 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A621 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A602 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A630 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A514 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A667 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A601 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A659 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A618 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A625 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A611 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A633 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A510 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A634 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN
#python src/utils/download_inmet_data.py -s A612 -b 2019 -e 2023 -o data/raw/inmet --api_token $INMET_API_TOKEN

# Concat inmet data
#python src/unify_inmet.py data/raw/inmet data/processed/inmet --aggregated True

# Calculate LST - OK
#python src/calculate_min_max_avg_lst.py 20191130 20231231 data/raw/lst data/processed/lst

# Download and Calculate RRQPE - OK
#python src/calculate_accumulated_rrqpe.py 20191130 20231231 data/raw/rrqpe data/processed/rrqpe

# Build
#python src/build_dataset.py data/processed/sinan/DENG.parquet data/processed/cnes/STRJ2401.parquet data/processed/sinan/sinan.parquet --start_date 2020-01-01 --end_date 2023-12-31

