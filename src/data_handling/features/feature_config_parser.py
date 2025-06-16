# src/data/feature_config_parser.py

import yaml

class FeatureConfig:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        self.weekly = config["dataset"]["weekly"]
        self.train_split = config["preproc"]["TRAIN_SPLIT"]
        self.val_split = config["preproc"]["VAL_SPLIT"]
        self.window_size = config["preproc"]["SLIDING_WINDOW_SIZE"]
        self.features = config["features"]