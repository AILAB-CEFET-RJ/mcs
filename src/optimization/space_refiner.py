# src/optimization/space_refiner.py

import numpy as np

def refine_space(study, old_space, shrink_factor=0.3, expand_ratio=1.3, boundary_trigger=0.95, min_shrink=0.05):
    """
    Refina dinamicamente o espaço de busca baseado nos trials já concluídos.
    
    Combina shrink dinâmico (centrado na mediana) com expansão de fronteira superior.
    """
    new_space = {}

    trials_df = study.trials_dataframe(attrs=("params", "value", "state"))
    trials_df = trials_df[trials_df["state"] == "COMPLETE"]

    for param_name in old_space.keys():
        if f"params_{param_name}" not in trials_df.columns:
            new_space[param_name] = old_space[param_name]
            continue

        values = trials_df[f"params_{param_name}"].dropna().astype(float).values

        if len(values) == 0:
            new_space[param_name] = old_space[param_name]
            continue

        old_low, old_high = old_space[param_name]

        median_val = np.median(values)
        shrink_amount = (old_high - old_low) * shrink_factor

        new_low = max(old_low, median_val - shrink_amount)
        new_high = min(old_high, median_val + shrink_amount)

        # Segurança mínima de shrink
        if (new_high - new_low) < (old_high - old_low) * min_shrink:
            new_low, new_high = old_low, old_high

        # Detector de boundary superior (expansão adaptativa)
        if np.max(values) >= old_high * boundary_trigger:
            new_high = old_high * expand_ratio

        # Segurança total: manter range válido
        if new_low >= new_high:
            new_low, new_high = old_low, old_high

        # Cast inteligente para int/float
        if isinstance(old_low, int) and isinstance(old_high, int):
            new_space[param_name] = (int(new_low), int(new_high))
        else:
            new_space[param_name] = (float(new_low), float(new_high))

    return new_space