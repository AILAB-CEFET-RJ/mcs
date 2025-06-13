# Treinamento em lote para todos os modelos e datasets (sem meteo, sem fração de zeros)

# === Caminhos principais ===
$SINAN_PATH = "data/processed/sinan/DENG.parquet"
$CNES_PATH = "data/processed/cnes/STRJ2401.parquet"
$CONFIG_PATH = "config/config.yaml"
$DATASET_DIR = "data/datasets"
$ERA5_PATH= "data/raw/era5/RJ_1997_2024.nc"
$RESULTS_DIR = "models/unoptmized"

# === Lista de nomes dos datasets ===
$weekly_map = @{
    "RJ" = $false
    "RJ_CASESONLY" = $false
    "RJ_WEEKLY" = $true
    "RJ_WEEKLY_CASESONLY" = $true
}

$casesonly_map = @{
    "RJ" = $false
    "RJ_CASESONLY" = $true
    "RJ_WEEKLY" = $false
    "RJ_WEEKLY_CASESONLY" = $true
}

$feacturesdict_map = @{
    "RJ" = "feature_dictionary.csv"
    "RJ_CASESONLY" = "feature_dictionary_cases.csv"
    "RJ_WEEKLY" = "feature_dictionary.csv"
    "RJ_WEEKLY_CASESONLY" = "feature_dictionary_cases.csv"
}

$scripts_map = @{
    "RJ" = ".py"
    "RJ_CASESONLY" = "_cases.py"
    "RJ_WEEKLY" = ".py"
    "RJ_WEEKLY_CASESONLY" = "_cases.py"
}

# === Modelos a treinar ===
$models = @(
    #"rf"
    #"xgb_poisson"
    "xgb_zip"
)

# === Seeds para replicabilidade ===
$seeds = @(0, 8, 109, 220, 222, 241, 149, 107, 75, 248)

# === Garantir que os diretórios existem ===
if (-not (Test-Path -Path $DATASET_DIR -PathType Container)) {
    New-Item -Path $DATASET_DIR -ItemType Directory -Force | Out-Null
}
if (-not (Test-Path -Path $RESULTS_DIR -PathType Container)) {
    New-Item -Path $RESULTS_DIR -ItemType Directory -Force | Out-Null
}

$build_dataset = $false

#=== Construção dos datasets ===
if ($build_dataset) {
    Write-Host "📦 Construindo datasets..."
    foreach ($dataset in $weekly_map.Keys) {
        $weekly = $weekly_map[$dataset]
        $script = $scripts_map[$dataset]
        $casesonly = $casesonly_map[$dataset]
        $out_path = Join-Path $DATASET_DIR "$dataset.pickle"

        Write-Host "🔧 $dataset (weekly=$weekly, casesonly=$casesonly)"
        python src/build_dataset.py `
            FULL `
            "$SINAN_PATH" `
            "$CNES_PATH" `
            "$ERA5_PATH" `
            "$out_path" `
            "$CONFIG_PATH" `
            "$weekly" `
            "$casesonly"
    }
}

# === Loop principal de treinamento ===
Write-Host "🎯 Iniciando treinamentos..."
foreach ($seed in $seeds) {
    foreach ($model in $models) {
        foreach ($dataset in $weekly_map.Keys) {            
            $dataset_path = Join-Path $DATASET_DIR "$dataset.pickle"
            $outdir = Join-Path $RESULTS_DIR "RJ_${seed}_${model}_$($dataset.ToLower())"
            $feacturesdict = $feacturesdict_map[$dataset]

            Write-Host "🚀 Treinando modelo '$model' no dataset '$dataset' com seed '$seed'..."
            if (-not (Test-Path -Path $outdir -PathType Container)) {
                New-Item -Path $outdir -ItemType Directory -Force | Out-Null
            }

            if (Test-Path -Path $dataset_path) {
                python "src/train_${model}.py" `
                    --dataset "$dataset_path" `
                    --outdir "$outdir" `
                    --seed "$seed" `
                    --dict "$feacturesdict"
            } else {
                Write-Warning "⚠️ Dataset não encontrado: $dataset_path. Pulando..."
            }
        }
    }
}

Write-Host "✅ Processo finalizado."
