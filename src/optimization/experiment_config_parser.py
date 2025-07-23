# src/optimization/experiment_config_parser.py

import yaml

class ExperimentConfig:
    def __init__(self, config_path):
        self.config_path = config_path
        self._load()

    def _load(self):
        with open(self.config_path, 'r') as f:
            self.raw_config = yaml.safe_load(f)

        # Carrega seções principais
        self.global_config = self.raw_config.get("global", {})
        self.experiments = self.raw_config.get("experiments", [])

        # Defaults globais
        self.seed = self.global_config.get("seed", 42)
        self.use_fixed_seed = self.global_config.get("use_fixed_seed", False)
        self.trials_per_round = self.global_config.get("trials_per_round", [50])
        self.runs_dir = self.global_config.get("runs_dir", "runs")
        self.directions = self.global_config.get("directions", ["minimize"])

        try:
            self.n_jobs_xgb = int(self.global_config.get("n_jobs_xgb", 8))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid n_jobs_xgb value in YAML: {self.global_config.get('n_jobs_xgb')}")
        
        self.early_stopping_rounds = self.global_config.get("early_stopping_rounds", 20)

    def list_experiments(self):
        """
        Retorna lista [(dataset_path, model_type)] para varrer experimentos.
        """
        pairs = []
        for exp in self.experiments:
            dataset = exp["dataset"]
            models = exp["models"]
            for model in models:
                pairs.append((dataset, model))
        return pairs
