@echo off
:: Build Datasets Script for Windows

:: Define paths to input data and configuration
set SINAN_PATH=data\processed\sinan\DENG.parquet
set CNES_PATH=data\processed\cnes\STRJ2401.parquet
set ERA5_PATH=data\raw\era5\RJ_1997_2024.nc
::set ERA5_PATH=data/processed/inmet/aggregated.parquet
set CONFIG_PATH=config\config.yaml
set CONCAT_PATH=data/processed/sinan/concat.parquet

::python legacy/src/unify_sinan.py data/raw/sinan data/processed/sinan
::python legacy/src/extract_sinan_cases.py %CONCAT_PATH% %SINAN_PATH% --filled --cod_uf 33 --start_date 2013-01-01 --end_date 2023-12-31

:: Run the Python script for each pipeline
::python src\build_dataset.py FULL %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\RJ.pickle %CONFIG_PATH% 
::python src\build_dataset_agg.py FULL %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\RJ_AGG.pickle %CONFIG_PATH% 
::python src\build_dataset_fixed.py FULL %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\RJ_FIX.pickle %CONFIG_PATH% 
::python src\build_dataset.py 7427549 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\7427549.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2268922 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2268922.pickle %CONFIG_PATH% 
::python src\build_dataset.py 7149328 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\7149328.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2299216 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2299216.pickle %CONFIG_PATH% 
::python src\build_dataset.py 0106453 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\0106453.pickle %CONFIG_PATH% 
::python src\build_dataset.py 6870066 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\6870066.pickle %CONFIG_PATH% 
::python src\build_dataset.py 6042619 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\6042619.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2288893 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2288893.pickle %CONFIG_PATH% 
::python src\build_dataset.py 5106702 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\5106702.pickle %CONFIG_PATH% 
::python src\build_dataset.py 6635148 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\6635148.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2269481 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2269481.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2708353 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2708353.pickle %CONFIG_PATH% 
::python src\build_dataset.py 7591136 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\7591136.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2283395 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2283395.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2287579 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2287579.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2291533 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2291533.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2292386 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2292386.pickle %CONFIG_PATH% 
::python src\build_dataset.py 0012505 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\0012505.pickle %CONFIG_PATH% 
::python src\build_dataset.py 2292084 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\2292084.pickle %CONFIG_PATH% 
::python src\build_dataset.py 6518893 %SINAN_PATH% %CNES_PATH% ERA5 %ERA5_PATH% data\datasets\6518893.pickle %CONFIG_PATH% 

:: Define the learner and task
set TASK=REGRESSION
set LEARNER=LstmNeuralNet

:: Train the model for each pipeline
::python src\train_model.py -t %TASK% -l %LEARNER% -p "FULL"
python src\train_randomforest.py --model xgboost --dataset data\datasets\2268922.pickle --outdir results
::python src\train_model.py -l %LEARNER% -p "7427549"
::python src\train_model.py -l %LEARNER% -p "2268922"
::python src\train_model.py -l %LEARNER% -p "7149328"
::python src\train_model.py -l %LEARNER% -p "2299216"
::python src\train_model.py -l %LEARNER% -p "0106453"
::python src\train_model.py -l %LEARNER% -p "6870066"
::python src\train_model.py -l %LEARNER% -p "6042619"
::python src\train_model.py -l %LEARNER% -p "2288893"
::python src\train_model.py -l %LEARNER% -p "5106702"
::python src\train_model.py -l %LEARNER% -p "6635148"
::python src\train_model.py -l %LEARNER% -p "2269481"
::python src\train_model.py -l %LEARNER% -p "2708353"
::python src\train_model.py -l %LEARNER% -p "7591136"
::python src\train_model.py -l %LEARNER% -p "2283395"
::python src\train_model.py -l %LEARNER% -p "2287579"
::python src\train_model.py -l %LEARNER% -p "2291533"
::python src\train_model.py -l %LEARNER% -p "2292386"
::python src\train_model.py -l %LEARNER% -p "0012505"
::python src\train_model.py -l %LEARNER% -p "2292084"
::python src\train_model.py -l %LEARNER% -p "6518893"

@echo on
