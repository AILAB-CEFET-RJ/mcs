@echo off

REM Set common parameters
set DATASET_PATH=data\processed\sinan\sinan.parquet
set OUTPUT_PATH=data\processed
set TRAIN_CUTOFF=2021-12-31
set VAL_CUTOFF=2022-12-31

REM List of id_unidade values
set ID_UNIDADES=7427549 2268922 7149328 2299216 0106453 6870066 6042619 2288893 5106702 6635148 2269481 2708353 7591136 2283395 2287579 2291533 2292386 0012505 2292084 6518893

REM Loop through each id_unidade
for %%I in (%ID_UNIDADES%) do (
    REM Train using XGBoost
    python train.py ^
        --dataset_path "%DATASET_PATH%" ^
        --output_path "%OUTPUT_PATH%\XGBOOST" ^
        --train_cutoff "%TRAIN_CUTOFF%" ^
        --val_cutoff "%VAL_CUTOFF%" ^
        --model_type "XGBoost" ^
        --id_unidade "%%I"

    REM Train using LSTM_PYTORCH
    python train.py ^
        --dataset_path "%DATASET_PATH%" ^
        --output_path "%OUTPUT_PATH%\LSTM_PYTORCH" ^
        --train_cutoff "%TRAIN_CUTOFF%" ^
        --val_cutoff "%VAL_CUTOFF%" ^
        --model_type "LSTM_PYTORCH" ^
        --id_unidade "%%I"
)
