#!/bin/bash
# Build Datasets Script for Linux

# Define paths to input data and configuration
SINAN_PATH=data/processed/sinan/DENG.parquet
CNES_PATH=data/processed/cnes/STRJ2401.parquet
ERA5_PATH=data/raw/era5/RJ_1997_2024.nc
CONFIG_PATH=config/config.yaml
CONCAT_PATH=data/processed/sinan/concat.parquet

mkdir -p data/datasets
mkdir -p results/rj_xgboost
mkdir -p results/rj_xgboost_fixed
mkdir -p results/rj_xgboost_weekly
mkdir -p results/rj_xgboost_weekly_fixed
mkdir -p results/rj_randomforest
mkdir -p results/rj_randomforest_fixed
mkdir -p results/rj_randomforest_weekly
mkdir -p results/rj_randomforest_weekly_fixed

# Pré-processamento comentado (descomente se necessário)
# python legacy/src/unify_sinan.py data/raw/sinan data/processed/sinan
# python legacy/src/extract_sinan_cases.py "$CONCAT_PATH" "$SINAN_PATH" --filled --cod_uf 33 --start_date 2013-01-01 --end_date 2023-12-31

# Build datasets com diferentes cenários
python src/build_dataset.py FULL "$SINAN_PATH" "$CNES_PATH" ERA5 "$ERA5_PATH" data/datasets/RJ.pickle "$CONFIG_PATH" none none false
python src/build_dataset.py FULL "$SINAN_PATH" "$CNES_PATH" ERA5 "$ERA5_PATH" data/datasets/RJ_FIXED.pickle "$CONFIG_PATH" -22.861389 -43.411389 false
python src/build_dataset.py FULL "$SINAN_PATH" "$CNES_PATH" ERA5 "$ERA5_PATH" data/datasets/RJ_WEEKLY.pickle "$CONFIG_PATH" none none true
python src/build_dataset.py FULL "$SINAN_PATH" "$CNES_PATH" ERA5 "$ERA5_PATH" data/datasets/RJ_WEEKLY_FIXED.pickle "$CONFIG_PATH" -22.861389 -43.411389 true

# Treinamento dos modelos XGBoost
python src/train_regressor.py --model xgboost --dataset data/datasets/RJ.pickle --outdir results/rj_xgboost
python src/train_regressor.py --model xgboost --dataset data/datasets/RJ_FIXED.pickle --outdir results/rj_xgboost_fixed
python src/train_regressor.py --model xgboost --dataset data/datasets/RJ_WEEKLY.pickle --outdir results/rj_xgboost_weekly
python src/train_regressor.py --model xgboost --dataset data/datasets/RJ_WEEKLY_FIXED.pickle --outdir results/rj_xgboost_weekly_fixed

# Treinamento dos modelos Random Forest
python src/train_regressor.py --model randomforest --dataset data/datasets/RJ.pickle --outdir results/rj_randomforest
python src/train_regressor.py --model randomforest --dataset data/datasets/RJ_FIXED.pickle --outdir results/rj_randomforest_fixed
python src/train_regressor.py --model randomforest --dataset data/datasets/RJ_WEEKLY.pickle --outdir results/rj_randomforest_weekly
python src/train_regressor.py --model randomforest --dataset data/datasets/RJ_WEEKLY_FIXED.pickle --outdir results/rj_randomforest_weekly_fixed
