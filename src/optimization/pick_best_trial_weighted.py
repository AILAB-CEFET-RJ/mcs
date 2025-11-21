import numpy as np
import pandas as pd

def pick_best_trial_weighted(self, study, weights=None, norm="minmax"):
    """
    Pick a single best trial from a multi-objective study
    using normalized weighted aggregation.

    All objectives are assumed to be minimized already.
    """

    # Get trials dataframe
    df = study.trials_dataframe()

    # Keep only COMPLETE trials
    df = df[df["state"] == "COMPLETE"].copy()

    value_cols = [c for c in df.columns if c.startswith("values_")]
    values = df[value_cols].to_numpy(dtype=float)

    # Default: equal weights
    m = values.shape[1]
    if weights is None:
        weights = np.ones(m) / m
    else:
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

    # ---- Normalize objectives ----
    if norm == "minmax":
        vmin = np.nanmin(values, axis=0)
        vmax = np.nanmax(values, axis=0)
        denom = np.where(vmax - vmin == 0, 1.0, vmax - vmin)
        values_norm = (values - vmin) / denom

    elif norm == "zscore":
        mu = np.nanmean(values, axis=0)
        sigma = np.nanstd(values, axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        values_norm = (values - mu) / sigma

    else:
        raise ValueError("norm must be 'minmax' or 'zscore'")

    # Weighted sum score
    scores = values_norm @ weights

    best_idx = int(np.nanargmin(scores))
    best_trial_number = int(df.iloc[best_idx]["number"])
    best_trial = study.trials[best_trial_number]

    return best_trial, scores[best_idx], df.iloc[best_idx]
