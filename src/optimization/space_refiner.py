# src/optimization/space_refiner.py

import numpy as np

CATEGORICAL_PARAMS = {"max_features", "bootstrap"}  # indices into mappings
NON_REFINABLE_PARAMS = {"model_seed"}  # optional but recommended

def refine_space(study, old_space, shrink_factor=0.3, expand_ratio=1.3,
                 boundary_trigger=0.95, min_shrink=0.05):

    new_space = {}

    trials_df = study.trials_dataframe(attrs=("params", "values", "state"))
    trials_df = trials_df[trials_df["state"] == "COMPLETE"]

    for param_name in old_space.keys():

        # Never refine categorical / seed params
        if param_name in CATEGORICAL_PARAMS or param_name in NON_REFINABLE_PARAMS:
            new_space[param_name] = old_space[param_name]
            continue

        col = f"params_{param_name}"
        if col not in trials_df.columns:
            new_space[param_name] = old_space[param_name]
            continue

        values = trials_df[col].dropna().astype(float).values
        if len(values) == 0:
            new_space[param_name] = old_space[param_name]
            continue

        old_low, old_high = old_space[param_name]

        median_val = np.median(values)
        shrink_amount = (old_high - old_low) * shrink_factor

        new_low = max(old_low, median_val - shrink_amount)
        new_high = min(old_high, median_val + shrink_amount)

        if (new_high - new_low) < (old_high - old_low) * min_shrink:
            new_low, new_high = old_low, old_high

        # expansion only for numeric params
        if np.max(values) >= old_high * boundary_trigger:
            new_high = old_high * expand_ratio

        if new_low >= new_high:
            new_low, new_high = old_low, old_high

        if "colsample" in param_name or "subsample" in param_name:
            new_low = max(0.0, min(new_low, 1.0))
            new_high = max(0.0, min(new_high, 1.0))

        if isinstance(old_low, int) and isinstance(old_high, int):
            new_space[param_name] = (int(new_low), int(new_high))
        else:
            new_space[param_name] = (float(new_low), float(new_high))

    return new_space
