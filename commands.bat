@echo off
:: Build Datasets Script for Windows

:: Define paths to input data and configuration
set SINAN_PATH=data\processed\sinan\DENG.parquet
set CNES_PATH=data\processed\cnes\STRJ2401.parquet
set ERA5_PATH=data\raw\era5\RJ_1997_2024.nc
set CONFIG_PATH=config\config.yaml
set CONCAT_PATH=data/processed/sinan/concat.parquet

::python legacy/src/unify_sinan.py data/raw/sinan data/processed/sinan
::python legacy/src/extract_sinan_cases.py %CONCAT_PATH% %SINAN_PATH% --filled --cod_uf 33 --start_date 2013-01-01 --end_date 2023-12-31

:: Run the Python script for each pipeline
python src\build_dataset.py FULL %SINAN_PATH% %CNES_PATH% %ERA5_PATH% data\datasets\RJ.pickle %CONFIG_PATH% none none false
python src\build_dataset.py FULL %SINAN_PATH% %CNES_PATH% %ERA5_PATH% data\datasets\RJ_FIXED.pickle %CONFIG_PATH% -22.861389 -43.411389 false
python src\build_dataset.py FULL %SINAN_PATH% %CNES_PATH% %ERA5_PATH% data\datasets\RJ_WEEKLY.pickle %CONFIG_PATH% none none true
python src\build_dataset.py FULL %SINAN_PATH% %CNES_PATH% %ERA5_PATH% data\datasets\RJ_WEEKLY_FIXED.pickle %CONFIG_PATH% -22.861389 -43.411389 true

:: Train the model for each pipeline
::python src\train_model.py -t %TASK% -l %LEARNER% -p "FULL"
python src\train_regressor.py --model xgboost --dataset data\datasets\RJ.pickle --outdir results/rj_xgboost
python src\train_regressor.py --model xgboost --dataset data\datasets\RJ_FIXED.pickle --outdir results/rj_xgboost_fixed
python src\train_regressor.py --model xgboost --dataset data\datasets\RJ_WEEKLY.pickle --outdir results/rj_xgboost_weekly
python src\train_regressor.py --model xgboost --dataset data\datasets\RJ_WEEKLY_FIXED.pickle --outdir results/rj_xgboost_weekly_fixed

python src\train_regressor.py --model randomforest --dataset data\datasets\RJ.pickle --outdir results/rj_randomforest
python src\train_regressor.py --model randomforest --dataset data\datasets\RJ_FIXED.pickle --outdir results/rj_randomforest_fixed
python src\train_regressor.py --model randomforest --dataset data\datasets\RJ_WEEKLY.pickle --outdir results/rj_randomforest_weekly
python src\train_regressor.py --model randomforest --dataset data\datasets\RJ_WEEKLY_FIXED.pickle --outdir results/rj_randomforest_weekly_fixed

@echo on
