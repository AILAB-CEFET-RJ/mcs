# src/optimization/experiment_runner.py

import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimization.adaptive_controller import run_adaptive_pipeline
from utils.utils import setup_logging, seed_everything
from optimization.config import SEED

if __name__ == "__main__":
    setup_logging()

    with open("config/experiments.yaml", "r") as f:
        config = yaml.safe_load(f)

    trials_list = config["global"]["trials_per_round"]
    seed = config["global"].get("seed", SEED)
    seed_everything(seed)

    for exp in config["experiments"]:
        dataset_path = exp["dataset"]
        models = exp["models"]

        for model_type in models:
            run_adaptive_pipeline(dataset_path, model_type, trials_list)
