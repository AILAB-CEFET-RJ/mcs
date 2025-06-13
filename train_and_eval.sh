#!/bin/bash
# Treinamento em lote para todos os modelos e datasets (com proporção de zeros ajustável)

# === Caminhos principais ===
SINAN_PATH=data/processed/sinan/DENG.parquet
CNES_PATH=data/processed/cnes/STRJ2401.parquet
ERA5_PATH=data/raw/era5/RJ_1997_2024.nc
CONFIG_PATH=config/config.yaml
DATASET_DIR=data/datasets
RESULTS_DIR=test

# === Lista de nomes dos datasets e proporções de zeros ===
declare -A zeroes_map=(
  [2268922]=1.0
  #[2268922_FIXED]=1.0
  [2268922_WEEKLY]=1.0
  #[2268922_WEEKLY_FIXED]=1.0
  [2268922_NO_ZEROES]=0.0
  #[2268922_FIXED_NO_ZEROES]=0.0
  [2268922_WEEKLY_NO_ZEROES]=0.0
  #[2268922_WEEKLY_FIXED_NO_ZEROES]=0.0
)

# === Se os dados são semanais (true/false) ===
declare -A weekly_map=(
  [2268922]=false
  #[2268922_FIXED]=false
  [2268922_WEEKLY]=true
  #[2268922_WEEKLY_FIXED]=true
  [2268922_NO_ZEROES]=false
  #[2268922_FIXED_NO_ZEROES]=false
  [2268922_WEEKLY_NO_ZEROES]=true
  #[2268922_WEEKLY_FIXED_NO_ZEROES]=true
)

# === Coordenadas fixas (ou none) ===
declare -A coord_map=(
  [2268922]="none none"
  #[2268922_FIXED]="-22.861389 -43.411389"
  [2268922_WEEKLY]="none none"
  #[2268922_WEEKLY_FIXED]="-22.861389 -43.411389"
  [2268922_NO_ZEROES]="none none"
  #[2268922_FIXED_NO_ZEROES]="-22.861389 -43.411389"
  [2268922_WEEKLY_NO_ZEROES]="none none"
  #[2268922_WEEKLY_FIXED_NO_ZEROES]="-22.861389 -43.411389"
)

# === Modelos a treinar ===
models=(
  randomforest
  xgboost
  xgboost_poisson
  xgboost_tweedie
)

# === Seeds para replicabilidade ===
#seeds=( 0 8 109 220 222 241 149 107 75 248 )
seeds=( 0 )

# === Garantir que os diretórios existem ===
mkdir -p "$DATASET_DIR"
mkdir -p "$RESULTS_DIR"

# === Construção dos datasets ===
echo "📦 Construindo datasets..."
for dataset in "${!zeroes_map[@]}"; do
  zeros="${zeroes_map[$dataset]}"
  weekly="${weekly_map[$dataset]}"
  coords=(${coord_map[$dataset]})
  lat="${coords[0]}"
  lon="${coords[1]}"
  out_path="$DATASET_DIR/$dataset.pickle"

  echo "🔧 $dataset (weekly=$weekly | zeros=$zeros | lat=$lat lon=$lon)"
  python src/build_dataset.py 2268922 "$SINAN_PATH" "$CNES_PATH" ERA5 "$ERA5_PATH" "$out_path" "$CONFIG_PATH" "$lat" "$lon" "$weekly" "$zeros"
done

# === Loop principal de treinamento ===
echo "🎯 Iniciando treinamentos..."
for seed in "${seeds[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${!zeroes_map[@]}"; do
      dataset_path="$DATASET_DIR/$dataset.pickle"
      outdir="$RESULTS_DIR/2268922_${seed}_${model}_${dataset,,}"

      echo "🚀 Treinando modelo '$model' no dataset '$dataset' com seed '$seed'..."
      mkdir -p "$outdir"

      if [[ -f "$dataset_path" ]]; then
        python src/train_regressor.py \
          --model "$model" \
          --dataset "$dataset_path" \
          --outdir "$outdir" \
          --seed "$seed"
      else
        echo "⚠️ Dataset não encontrado: $dataset_path. Pulando..."
      fi
    done
  done
done

echo "✅ Processo finalizado."