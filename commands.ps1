# Treinamento em lote para todos os modelos e datasets (com proporção de zeros ajustável)

# === Caminhos principais ===
$SINAN_PATH = "data/processed/sinan/DENG.parquet"
$CNES_PATH = "data/processed/cnes/STRJ2401.parquet"
$ERA5_PATH = "data/raw/era5/RJ_1997_2024.nc"
$CONFIG_PATH = "config/config.yaml"
$DATASET_DIR = "data/datasets"
$RESULTS_DIR = "test"

# === Lista de nomes dos datasets e proporções de zeros ===
$zeroes_map = @{
    "RJ" = 1.0
    "RJ_WEEKLY" = 1.0
    "RJ_NO_ZEROES" = 0.0
    "RJ_WEEKLY_NO_ZEROES" = 0.0
}

# === Se os dados são semanais (true/false) ===
$weekly_map = @{
    "RJ" = $false
    "RJ_WEEKLY" = $true
    "RJ_NO_ZEROES" = $false
    "RJ_WEEKLY_NO_ZEROES" = $true
}

# === Coordenadas fixas (ou none) ===
$coord_map = @{
    "RJ" = "none none"
    "RJ_WEEKLY" = "none none"
    "RJ_NO_ZEROES" = "none none"
    "RJ_WEEKLY_NO_ZEROES" = "none none"
}

# === Modelos a treinar ===
$models = @(
    #"randomforest"
    "xgboost"
    "xgboost_poisson"
    "xgboost_tweedie"
)

# === Seeds para replicabilidade ===
$seeds = @(0)
#$seeds = @(0, 8, 109, 220, 222, 241, 149, 107, 75, 248)

# === Garantir que os diretórios existem ===
if (-not (Test-Path -Path $DATASET_DIR -PathType Container)) {
    New-Item -Path $DATASET_DIR -ItemType Directory -Force | Out-Null
}
if (-not (Test-Path -Path $RESULTS_DIR -PathType Container)) {
    New-Item -Path $RESULTS_DIR -ItemType Directory -Force | Out-Null
}

# === Construção dos datasets ===
Write-Host "📦 Construindo datasets..."
foreach ($dataset in $zeroes_map.Keys) {
    $zeros = $zeroes_map[$dataset]
    $weekly = $weekly_map[$dataset]
    $coords = $coord_map[$dataset].Split(" ")
    $lat = $coords[0]
    $lon = $coords[1]
    $out_path = Join-Path $DATASET_DIR "$dataset.pickle"

    Write-Host "🔧 $dataset (weekly=$weekly | zeros=$zeros | lat=$lat lon=$lon)"
    python src/build_dataset.py FULL "$SINAN_PATH" "$CNES_PATH" ERA5 "$ERA5_PATH" "$out_path" "$CONFIG_PATH" "$lat" "$lon" "$weekly" "$zeros"
}

# === Loop principal de treinamento ===
Write-Host "🎯 Iniciando treinamentos..."
foreach ($seed in $seeds) {
    foreach ($model in $models) {
        foreach ($dataset in $zeroes_map.Keys) {
            $dataset_path = Join-Path $DATASET_DIR "$dataset.pickle"
            $outdir = Join-Path $RESULTS_DIR "RJ_${seed}_${model}_$($dataset.ToLower())"

            Write-Host "🚀 Treinando modelo '$model' no dataset '$dataset' com seed '$seed'..."
            if (-not (Test-Path -Path $outdir -PathType Container)) {
                New-Item -Path $outdir -ItemType Directory -Force | Out-Null
            }

            if (Test-Path -Path $dataset_path) {
                python src/train_regressor.py `
                    --model "$model" `
                    --dataset "$dataset_path" `
                    --outdir "$outdir" `
                    --seed "$seed"
            } else {
                Write-Warning "⚠️ Dataset não encontrado: $dataset_path. Pulando..."
            }
        }
    }
}

Write-Host "✅ Processo finalizado."