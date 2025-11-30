# src/optimization/space_refiner.py

import numpy as np

# Heurística simples pra detectar params em escala log
LOG_PARAMS_HINTS = ("learning_rate", "reg_alpha", "reg_lambda")

def refine_space(
    study,
    old_space,
    weights=None,
    norm="minmax",
    top_frac=0.2,
    shrink_factor=0.3,
    expand_ratio=1.3,
    boundary_trigger=0.95,
    min_shrink=0.05
):
    """
    Refina dinamicamente o espaço de busca baseado nos trials já concluídos,
    usando agregação ponderada multi-objetivo PARA ESCOLHER apenas os melhores trials
    e então encolher o espaço em torno da mediana deles.

    Premissas:
      - Todos os objetivos já estão em modo "minimize".
      - study é multiobjetivo (values_0, values_1, ...).

    Parâmetros:
      weights: lista/np.array de pesos p/ objetivos (opcional). Se None, pesos iguais.
      norm: "minmax" ou "zscore"
      top_frac: fração dos melhores trials usados para refino (ex: 0.2 = top 20%)
      shrink_factor: quanto do range antigo será usado pra encolher ao redor da mediana
      expand_ratio: quanto expandir quando bater borda
      boundary_trigger: quão perto da borda (percentual) para disparar expansão
      min_shrink: mínimo percentual de range (se ficar menor, reverte ao range antigo)
    """

    df = study.trials_dataframe(attrs=("params", "values", "state"))
    df = df[df["state"] == "COMPLETE"].copy()

    # se não tem trial completo, não refina
    if len(df) == 0:
        return old_space

    # --------- 1) Score ponderado multi-objetivo ---------
    value_cols = [c for c in df.columns if c.startswith("values_")]
    V = df[value_cols].to_numpy(dtype=float)

    m = V.shape[1]
    if weights is None:
        w = np.ones(m, dtype=float) / m
    else:
        w = np.array(weights, dtype=float)
        w = w / w.sum()

    # Normalização
    if norm == "minmax":
        vmin = np.nanmin(V, axis=0)
        vmax = np.nanmax(V, axis=0)
        denom = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
        Vn = (V - vmin) / denom

    elif norm == "zscore":
        mu = np.nanmean(V, axis=0)
        sigma = np.nanstd(V, axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        Vn = (V - mu) / sigma

    else:
        raise ValueError("norm must be 'minmax' or 'zscore'")

    scores = Vn @ w
    df["weighted_score"] = scores

    # --------- 2) Seleciona top_frac melhores trials ---------
    k = max(1, int(len(df) * top_frac))
    best_df = df.nsmallest(k, "weighted_score")

    new_space = {}

    for param_name, (old_low, old_high) in old_space.items():
        col = f"params_{param_name}"

        # se o param nem apareceu nos trials, mantém
        if col not in best_df.columns:
            new_space[param_name] = (old_low, old_high)
            continue

        vals = best_df[col].dropna().astype(float).values
        if len(vals) == 0:
            new_space[param_name] = (old_low, old_high)
            continue

        # --------- 3) Refinar no espaço certo (log ou linear) ---------
        is_log = any(h in param_name for h in LOG_PARAMS_HINTS)

        if is_log:
            # protege contra zeros/negativos
            safe_vals = np.clip(vals, 1e-12, None)
            vals_t = np.log10(safe_vals)

            old_low_t = np.log10(max(old_low, 1e-12))
            old_high_t = np.log10(max(old_high, 1e-12))
        else:
            vals_t = vals
            old_low_t = float(old_low)
            old_high_t = float(old_high)

        # mediana dos melhores
        med_t = np.median(vals_t)

        # encolhe ao redor da mediana
        shrink_amount = (old_high_t - old_low_t) * shrink_factor
        new_low_t  = max(old_low_t,  med_t - shrink_amount)
        new_high_t = min(old_high_t, med_t + shrink_amount)

        # segurança mínima de shrink
        if (new_high_t - new_low_t) < (old_high_t - old_low_t) * min_shrink:
            new_low_t, new_high_t = old_low_t, old_high_t

        # --------- 4) Expansão adaptativa de fronteiras ---------
        best_max_t = np.max(vals_t)
        best_min_t = np.min(vals_t)

        high_trigger_t = old_high_t * boundary_trigger
        low_trigger_t  = old_low_t * (2 - boundary_trigger)  # simétrico

        if best_max_t >= high_trigger_t:
            new_high_t = old_high_t * expand_ratio

        if best_min_t <= low_trigger_t:
            new_low_t = old_low_t / expand_ratio

        # mantém range válido
        if new_low_t >= new_high_t:
            new_low_t, new_high_t = old_low_t, old_high_t

        # volta do log-space
        if is_log:
            new_low  = 10 ** new_low_t
            new_high = 10 ** new_high_t
        else:
            new_low, new_high = new_low_t, new_high_t

        # clamp para [0,1]
        if "colsample" in param_name or "subsample" in param_name:
            new_low = max(0.0, min(new_low, 1.0))
            new_high = max(0.0, min(new_high, 1.0))

        # cast inteligente
        if isinstance(old_low, int) and isinstance(old_high, int):
            new_space[param_name] = (int(round(new_low)), int(round(new_high)))
        else:
            new_space[param_name] = (float(new_low), float(new_high))

    return new_space
