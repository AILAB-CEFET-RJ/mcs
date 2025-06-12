# === utils.py ===

import os
import pandas as pd
import logging
from optuna.visualization import plot_optimization_history, plot_param_importances

# Setup global logger

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

# Garantir reprodutibilidade completa

def seed_everything(seed):
    import numpy as np
    import random
    import os
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass

# Salvar resultados de forma padronizada

# Salvamento completo dos resultados de estudo
def save_study_results(study, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Salva melhor trial
    pd.Series(study.best_trial.params).to_json(os.path.join(output_dir, "best_params.json"), indent=4)

    # Salva todos os trials com parâmetros e métricas completas
    all_data = []

    for t in study.trials:
        row = {"trial": t.number, "value": t.value}

        # Parametros
        for k, v in t.params.items():
            row[f"param_{k}"] = v

        # Métricas customizadas (user_attrs)
        if "metrics" in t.user_attrs:
            for k, v in t.user_attrs["metrics"].items():
                row[f"metric_{k}"] = v

        all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(output_dir, "trials.csv"), index=False)

    # Plots (opcionais)
    try:
        plot_optimization_history(study).write_html(os.path.join(output_dir, "opt_history.html"))
        plot_param_importances(study).write_html(os.path.join(output_dir, "opt_importance.html"))
    except Exception as e:
        logging.warning(f"Falha ao gerar visualizações: {e}")

    logging.info(f"Resultados completos salvos em {output_dir}")
