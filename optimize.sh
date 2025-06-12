#!/bin/bash

# === Caminhos principais ===
DATASET_DIR="data/datasets"
OPTUNA_PIPELINE="src/optimization/optimize_pipeline.py"
RESULTS_DIR="runs"

# === Lista de nomes dos datasets ===
declare -A weekly_map=(
    ["RJ"]=false
    ["RJ_CASESONLY"]=false
    ["RJ_WEEKLY"]=true
    ["RJ_WEEKLY_CASESONLY"]=true
)

# === Mapear datasets em pickle ===
datasets=()
for dataset in "${!weekly_map[@]}"; do
    path="$DATASET_DIR/$dataset.pickle"
    if [[ -f "$path" ]]; then
        datasets+=("$path")
    else
        echo "⚠️ Dataset não encontrado: $path"
    fi
done

# === Modelos para otimização ===
models=("poisson" "zip" "rf")

# === Trials por estudo ===
trials=50

# === Loop principal ===
for dataset_path in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "🚀 Otimizando modelo '$model' para dataset '$dataset_path'..."
        python "$OPTUNA_PIPELINE" \
            --datasets "$dataset_path" \
            --models "$model" \
            --trials "$trials"
    done
done

echo "✅ Processo de otimização finalizado."
