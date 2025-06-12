# === config.py ===

# Configurações globais de execução

N_JOBS_XGB = 8
N_TRIALS_DEFAULT = 50
SEED = 987
USE_FIXED_SEED = False

# Diretórios
RUNS_DIR = "runs"

# Early stopping
EARLY_STOPPING_ROUNDS = 20

# Trial objective directions
DIRECTIONS = [
    "minimize",  # MSE
    "minimize",  # RMSE
    "minimize",  # MAE
    "minimize",  # -R²
    "minimize",  # MAPE
    "minimize",  # SMAPE
    "minimize",  # -Rho
    "minimize"   # Poisson Deviance
]