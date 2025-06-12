# Arboseer - Adaptive Dengue Forecasting Pipeline

AutoML Adaptive Hyperparameter Optimization + Full Data Pipeline for Arbovirus Forecasting

---

## 📌 Descrição Geral

O Arboseer é um pipeline científico completo, modular e reprodutível para:

- Download e ingestão de dados brutos (SINAN, CNES, ERA5)
- Pré-processamento e limpeza de dados
- Construção de datasets prontos para modelagem
- Treinamento tradicional de modelos
- Otimização adaptativa de hiperparâmetros (AutoML)
- Registro completo de métricas de desempenho

---

## 📂 Estrutura de Diretórios

.
├── config/           
│   ├── config.yaml
│   ├── environment.yml
│   └── experiments.yaml
│
├── data/             
│   ├── raw/
│   ├── processed/
│   └── datasets/
│
├── ingestion/        
│   ├── download/
│   │   ├── download_cnes.py
│   │   └── download_sinan.py
│   └── preprocess/
│       ├── preprocess_cnes.py
│       └── preprocess_sinan.py
│
├── models/           
├── notebooks/        
├── runs/             
├── src/              
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── optimization/
│   ├── utils/
│   └── train_xgb_poisson.py, train_xgb_zip.py, train_rf.py
│
├── tests/            
├── README.md         
└── environment.yml   

---

# 🚀 Fluxo 0 — Ingestão e Pré-Processamento

## 0️⃣ Download dos dados brutos

Os dados são baixados diretamente pelos scripts em `ingestion/download/`:

### SINAN:
python src/ingestion/download/download_sinan.py DENG 2023 data/raw/sinan/

### CNES:
python src/ingestion/download/download_cnes.py ST RJ 2311 data/raw/cnes/STRJ2311.dbc

> **Nota:** os parâmetros devem ser ajustados conforme o período e o UF desejados.

## 0️⃣ Pré-processamento inicial

Após o download, os dados são padronizados e limpos via:

### SINAN:
python src/ingestion/preprocess/preprocess_sinan.py data/raw/sinan/ data/processed/sinan/

### CNES:
python src/ingestion/preprocess/preprocess_cnes.py data/raw/cnes/STRJ2311.parquet data/processed/cnes/STRJ2401.parquet

> O ERA5 já vem pré-processado.

---

# 🚀 Fluxo 1 — Construção dos Datasets

A construção final dos datasets de modelagem ocorre via:

python src/data/build_dataset.py \
    FULL \
    data/processed/sinan/DENG.parquet \
    data/processed/cnes/STRJ2401.parquet \
    data/raw/era5/RJ_1997_2024.nc \
    data/datasets/RJ_WEEKLY.pickle \
    config/config.yaml \
    True  True

- Integra dados temporais e espaciais
- Constrói janelas deslizantes
- Exporta o dataset no formato pickle

---

# 🚀 Fluxo 2 — Treinamento Tradicional de Modelos

Treinamento manual (sem otimização adaptativa):

### Poisson:

python src/train_xgb_poisson.py \
  --dataset data/datasets/RJ_WEEKLY.pickle \
  --outdir runs/teste_poisson \
  --seed 42 \
  --dict feature_dictionary.csv

### ZIP:

python src/train_xgb_zip.py \
  --dataset data/datasets/RJ_WEEKLY.pickle \
  --outdir runs/teste_zip \
  --seed 42 \
  --dict feature_dictionary.csv

### Random Forest:

python src/train_rf.py \
  --dataset data/datasets/RJ_WEEKLY.pickle \
  --outdir runs/teste_rf \
  --seed 42 \
  --dict feature_dictionary.csv

Cada execução:

- Treina o modelo
- Calcula métricas de treino, validação e teste
- Gera curvas de aprendizado, distribuição de predições e feature importance

---

# 🚀 Fluxo 3 — Pipeline de Otimização Adaptativa (AutoML)

## 1️⃣ Definir experimentos no YAML:

Arquivo `config/experiments.yaml`:

global:
  trials_per_round: [50, 100, 150]
  seed: 42

experiments:
  - dataset: data/datasets/RJ.pickle
    models: [poisson, zip, rf]

  - dataset: data/datasets/RJ_WEEKLY.pickle
    models: [poisson, zip]

## 2️⃣ Executar o pipeline completo:

python src/optimization/experiment_runner.py

O pipeline irá:

- Rodar múltiplos datasets e modelos automaticamente
- Aplicar o Adaptive Search Space Refinement
- Registrar resultados de cada experimento

---

# 📊 Resultados Gerados

Todos os resultados são organizados em `runs/`, contendo:

- best_params.json → melhores hiperparâmetros
- trials.csv → histórico completo de trials
- opt_history.html → curva de otimização
- opt_importance.html → importância dos hiperparâmetros

---

# 📈 Métricas Calculadas

- MSE (Mean Squared Error)
- RMSE
- MAE
- R²
- MAPE (ignorando zeros)
- SMAPE
- Poisson Deviance
- Pearson Correlation (ρ)

---

# ⚙️ Instalação do Ambiente

Pré-requisito: Conda

Clone o projeto:

git clone <REPO_URL>
cd <REPO_DIR>

Crie o ambiente:

conda env create -f config/environment.yml

Ative o ambiente:

conda activate arboseer

(Recomendado) Exportar novamente o ambiente:

conda env export --from-history > config/environment.yml

---

# 📦 Dependências Principais

- numpy
- pandas
- scikit-learn
- xgboost
- optuna
- pyyaml
- matplotlib
- joblib
- torch
- tqdm

---