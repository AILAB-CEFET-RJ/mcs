# Treinamento em lote para todos os modelos e datasets (sem meteo, sem fração de zeros)

# === Caminhos principais ===
$SINAN_PATH = "data/processed/sinan/DENG.parquet"
$CNES_PATH = "data/processed/cnes/STRJ2401.parquet"
$CONFIG_PATH = "config/config.yaml"
$DATASET_DIR = "data/datasets"
$ERA5_PATH= "data/raw/era5/RJ_1997_2024.nc"
$RESULTS_DIR = "test_yescases_unoptmized"

# === Lista de nomes dos datasets ===
$weekly_map = @{
    "RJ" = $false
    "RJ_WEEKLY" = $true
}

# === Modelos a treinar ===
$models = @(
    #"randomforest"
    "xgboost_poisson"
)

# === Seeds para replicabilidade ===
#$seeds = @(0, 8, 109, 220, 222, 241, 149, 107, 75, 248)
$seeds = @(0)

# === Garantir que os diretórios existem ===
if (-not (Test-Path -Path $DATASET_DIR -PathType Container)) {
    New-Item -Path $DATASET_DIR -ItemType Directory -Force | Out-Null
}
if (-not (Test-Path -Path $RESULTS_DIR -PathType Container)) {
    New-Item -Path $RESULTS_DIR -ItemType Directory -Force | Out-Null
}

# === Construção dos datasets ===
Write-Host "📦 Construindo datasets..."
foreach ($dataset in $weekly_map.Keys) {
    $weekly = $weekly_map[$dataset]
    $out_path = Join-Path $DATASET_DIR "$dataset.pickle"

    Write-Host "🔧 $dataset (weekly=$weekly)"
    python src/build_dataset.py `
        FULL `
        "$SINAN_PATH" `
        "$CNES_PATH" `
        ERA5 `
        "$ERA5_PATH" `
        "$out_path" `
        "$CONFIG_PATH" `
        "$weekly"
}

# === Loop principal de treinamento ===
Write-Host "🎯 Iniciando treinamentos..."
foreach ($seed in $seeds) {
    foreach ($model in $models) {
        foreach ($dataset in $weekly_map.Keys) {
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
