# src/optimization/space_refiner.py

import numpy as np

# Função de refinamento adaptativo do espaço de busca
def refine_space(study, old_space, expand_ratio=1.3, shrink_percentile=20):
    """
    Refina dinamicamente o espaço de busca a partir do histórico de trials.
    """
    new_space = {}

    # Obtem apenas trials bem sucedidos
    trials = [t for t in study.trials if t.state == 'COMPLETE']

    for param_name in old_space.keys():
        values = np.array([
            t.params[param_name] for t in trials 
            if param_name in t.params
        ])

        if len(values) == 0:
            # Parâmetro ainda não foi sugerido
            new_space[param_name] = old_space[param_name]
            continue

        low, high = old_space[param_name]

        # Detecção de boundary superior
        boundary_expand = False
        if np.max(values) >= high * 0.95:
            boundary_expand = True

        # Refinar limites inferior e superior
        refined_low = max(low, np.percentile(values, shrink_percentile))
        refined_high = min(high, np.percentile(values, 100 - shrink_percentile))

        if boundary_expand:
            refined_high = int(high * expand_ratio)

        # Garantir que não encolha demais
        if refined_high - refined_low < 2:
            refined_low, refined_high = low, high

        new_space[param_name] = (int(refined_low), int(refined_high))

    return new_space