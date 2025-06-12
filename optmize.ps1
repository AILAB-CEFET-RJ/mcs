# === Caminhos principais ===
$DATASET_DIR = "data/datasets"
$OPTUNA_PIPELINE = "src/optmization/optmize_pipeline.py"
$RESULTS_DIR = "runs"

# === Lista de nomes dos datasets ===
$weekly_map = @{
    "RJ" = $false
    "RJ_CASESONLY" = $false
    "RJ_WEEKLY" = $true
    "RJ_WEEKLY_CASESONLY" = $true
}

# === Mapear datasets em pickle ===
$datasets = @()
foreach ($dataset in $weekly_map.Keys) {
    $path = Join-Path $DATASET_DIR "$dataset.pickle"
    if (Test-Path -Path $path) {
        $datasets += $path
    } else {
        Write-Warning "⚠️ Dataset não encontrado: $path"
    }
}

# === Modelos para otimização ===
$models = @("poisson", "zip", "rf")

# === Trials por estudo ===
$trials = 50

# === Loop principal ===
foreach ($dataset_path in $datasets) {
    foreach ($model in $models) {
        Write-Host "🚀 Otimizando modelo '$model' para dataset '$dataset_path'..."
        python $OPTUNA_PIPELINE `
            --datasets $dataset_path `
            --models $model `
            --trials $trials
    }
}

Write-Host "✅ Processo de otimização finalizado."
