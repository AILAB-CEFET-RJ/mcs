@echo off
REM Set common parameters
set DATASET_PATH=data\processed\sinan\sinan.parquet
set DATE_START=2023-01-01
set DATE_END=2023-12-31

REM Directories for models and results
set XGBOOST_MODELS_DIR=data\processed\XGBOOST
set LSTM_PYTORCH_MODELS_DIR=data\processed\LSTM_PYTORCH
set XGBOOST_RESULTS_DIR=data\processed\XGBOOST_RESULTS
set LSTM_PYTORCH_RESULTS_DIR=data\processed\LSTM_PYTORCH_RESULTS

REM List of ID_UNIDADE values
set ID_UNIDADES=7427549 2268922 7149328 2299216 0106453 6870066 6042619 2288893 5106702 6635148 2269481 2708353 7591136 2283395 2287579 2291533 2292386 0012505 2292084 6518893

REM List of feature sets
set FEATURE_SETS=all_features cases

REM Create result directories if they don't exist
if not exist "%XGBOOST_RESULTS_DIR%" mkdir "%XGBOOST_RESULTS_DIR%"
if not exist "%LSTM_PYTORCH_RESULTS_DIR%" mkdir "%LSTM_PYTORCH_RESULTS_DIR%"

REM Loop through each ID_UNIDADE
for %%I in (%ID_UNIDADES%) do (
    REM Loop through each feature set
    for %%F in (%FEATURE_SETS%) do (
        REM Process XGBoost model
        if exist "%XGBOOST_MODELS_DIR%\XGBoost_Model_%%I_%%F.json" (
            echo Processing XGBoost model for ID_UNIDADE: %%I and feature set: %%F
            python predict.py ^
                --dataset_path "%DATASET_PATH%" ^
                --model_path "%XGBOOST_MODELS_DIR%\XGBoost_Model_%%I_%%F.json" ^
                --id_unidade "%%I" ^
                --feature_set_name "%%F" ^
                --date_start "%DATE_START%" ^
                --date_end "%DATE_END%" ^
                --model_type "XGBoost" ^
                --output_path "%XGBOOST_RESULTS_DIR%\%%I_%%F"
        ) else (
            echo XGBoost model for ID_UNIDADE %%I and feature set %%F not found, skipping...
        )

        REM Process LSTM_PyTorch model
        if exist "%LSTM_PYTORCH_MODELS_DIR%\LSTM_PyTorchModel_%%I_%%F.pt" (
            echo Processing LSTM_PyTorch model for ID_UNIDADE: %%I and feature set: %%F
            python predict.py ^
                --dataset_path "%DATASET_PATH%" ^
                --model_path "%LSTM_PYTORCH_MODELS_DIR%\LSTM_PyTorchModel_%%I_%%F.pt" ^
                --id_unidade "%%I" ^
                --feature_set_name "%%F" ^
                --date_start "%DATE_START%" ^
                --date_end "%DATE_END%" ^
                --model_type "LSTM_PYTORCH" ^
                --output_path "%LSTM_PYTORCH_RESULTS_DIR%\%%I_%%F"
        ) else (
            echo LSTM_PyTorch model for ID_UNIDADE %%I and feature set %%F not found, skipping...
        )
    )
)
