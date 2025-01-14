import pandas as pd
import os
import logging
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Abstract Base Model
class BaseModel:
    def __init__(self, name, scaler=MinMaxScaler(), batch_size=32, epochs=100):
        self.name = name
        self.scaler = scaler
        self.batch_size = batch_size
        self.epochs = epochs

    def prepare_data(self, X_train, X_val):
        X_train_scaled, X_val_scaled = self.scale_data(X_train, X_val)
        return self.reshape_data(X_train_scaled, X_val_scaled)

    def scale_data(self, X_train, X_val):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        return X_train_scaled, X_val_scaled

    def reshape_data(self, X_train_scaled, X_val_scaled):
        return X_train_scaled, X_val_scaled

    def train_and_save(self, data, config):
        raise NotImplementedError("train_and_save must be implemented in subclasses")


# Dataset Class for PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# PyTorch LSTM Model Class
class LSTM_PyTorchModel(BaseModel):
    def __init__(self, input_size, hidden_size=50, batch_size=32, epochs=100, name="LSTM_PyTorchModel"):
        super().__init__(name, batch_size=batch_size, epochs=epochs)
        self.model = torch.nn.Sequential(
            torch.nn.LSTM(input_size, hidden_size, batch_first=True),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.ReLU()  # Ensures non-negative outputs
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def save_model(self, output_path, unit_id, feature_set_name):
        model_path = os.path.join(output_path, f"{self.name}_{unit_id}_{feature_set_name}.pt")
        os.makedirs(output_path, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")

    def train_and_save(self, data, config):
        X_train, y_train, X_val, y_val = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
        output_path, unit_id, feature_set_name = config["output_path"], config["unit_id"], config["feature_set_name"]

        X_train, X_val = self.prepare_data(X_train, X_val)

        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)

        history = {"loss": [], "val_loss": []}
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs, _ = self.model[0](batch_X)
                outputs = self.model[1](outputs.squeeze(1))
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            history["loss"].append(epoch_loss / len(train_loader.dataset))

            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs, _ = self.model[0](batch_X)
                    outputs = self.model[1](outputs.squeeze(1))
                    outputs = torch.relu(outputs)  # Ensure non-negative
                    outputs = torch.round(outputs)  # Convert to integers
                    val_loss += self.criterion(outputs, batch_y).item() * batch_X.size(0)
            history["val_loss"].append(val_loss / len(val_loader.dataset))

            logging.info(f'Epoch {epoch+1}/{self.epochs} | Train Loss: {history["loss"][-1]:.4f} | Val Loss: {history["val_loss"][-1]:.4f}')

        self.save_loss_plot(history, output_path, unit_id, feature_set_name)
        self.save_model(output_path, unit_id, feature_set_name)

    def save_loss_plot(self, history, output_path, unit_id, feature_set_name):
        plt.figure(figsize=(10, 6))
        plt.plot(history["loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title(f"Train vs Validation Loss for {unit_id} - {feature_set_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_path, f"loss_plot_{unit_id}_{feature_set_name}.png")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Loss plot saved to {plot_path}")


class XGBoostModel(BaseModel):
    def __init__(self, name="XGBoost_Model"):
        super().__init__(name)
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=128, learning_rate=0.1)

    def save_model(self, output_path, unit_id, feature_set_name):
        model_path = os.path.join(output_path, f"{self.name}_{unit_id}_{feature_set_name}.json")
        os.makedirs(output_path, exist_ok=True)
        self.model.save_model(model_path)
        logging.info(f"Model saved to {model_path}")

    def save_plot(self, X_val, y_val, predictions, output_path, unit_id, feature_set_name):
        """
        Saves a plot comparing actual vs predicted values.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_val, label="Actual", marker="o")
        plt.plot(predictions, label="Predicted", marker="x")
        plt.title(f"Predicted vs Actual for Unit: {unit_id}, Features: {feature_set_name}")
        plt.xlabel("Samples")
        plt.ylabel("Cases")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(output_path, f"plot_{unit_id}_{feature_set_name}.png")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Plot saved to {plot_path}")

    def train_and_save(self, data, config):
        """
        Trains the XGBoost model, saves it, and generates a plot.
        """
        X_train, y_train, X_val, y_val = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
        output_path, unit_id, feature_set_name = config["output_path"], config["unit_id"], config["feature_set_name"]

        # Prepare data
        X_train, X_val = self.prepare_data(X_train, X_val)

        # Train the model
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.model.set_params(early_stopping_rounds=10)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

        # Make predictions
        predictions = self.model.predict(X_val)
        predictions = np.clip(np.round(predictions), 0, None)  # Non-negative integers

        # Save the model
        self.save_model(output_path, unit_id, feature_set_name)

        # Save the plot
        self.save_plot(X_val, y_val, predictions, output_path, unit_id, feature_set_name)


def train(dataset_path, output_path, train_cutoff, val_cutoff, model_type, id_unidade=None):
    """
    Train models on the dataset for disease case predictions.

    Args:
        dataset_path (str): Path to the dataset.
        output_path (str): Directory to save trained models and outputs.
        train_cutoff (str): Date cutoff for training data.
        val_cutoff (str): Date cutoff for validation data.
        model_type (str): Type of model to train ('LSTM_PYTORCH' or 'XGBoost').
        id_unidade (int or None): Specific unit ID for training. If None, trains on all units.

    Returns:
        None
    """
    # Load dataset
    df = pd.read_parquet(dataset_path)
    if id_unidade:
        df = df[df["ID_UNIDADE"] == id_unidade]

    # Feature groups
    features_cases = ['CASES', 'CASES_MM_14', 'CASES_MM_21', 'CASES_ACC_14', 'CASES_ACC_21']
    windows = [7, 14, 21]
    features_era5 = ['TEM_MIN_ERA5', 'TEM_AVG_ERA5', 'TEM_MAX_ERA5',
                     'IDEAL_TEMP_ERA5', 'EXTREME_TEMP_ERA5', 'TEMP_RANGE_ERA5'] + [
                    f'TEM_AVG_ERA5_MM_{window}' for window in windows] + [
                    f'TEMP_RANGE_ERA5_MM_{window}' for window in windows] + [
                    f'TEM_AVG_ERA5_ACC_{window}' for window in windows]

    all_features = features_cases + features_era5
    training_features = {
        "cases": features_cases,
        "all_features": all_features
    }

    target = "CASES"

    # Drop missing values
    df = df.dropna()

    # Group by unit if applicable
    grouped = df.groupby("ID_UNIDADE")

    # Select model class
    if model_type == "LSTM_PYTORCH":
        ModelClass = LSTM_PyTorchModel
    elif model_type == "XGBoost":
        ModelClass = XGBoostModel
    else:
        raise ValueError("Invalid model type! Use 'LSTM_PYTORCH' or 'XGBoost'.")

    for name, group in grouped:
        group = group.sort_values(by="DT_NOTIFIC")
        train_df = group[group["DT_NOTIFIC"] <= train_cutoff]
        val_df = group[(group["DT_NOTIFIC"] > train_cutoff) & (group["DT_NOTIFIC"] <= val_cutoff)]

        for feature_set_name, feature_set in training_features.items():
            # Skip missing feature sets
            missing_features = [f for f in feature_set if f not in df.columns]
            if missing_features:
                logging.warning(f"Skipping feature set {feature_set_name} for {name}: Missing features {missing_features}")
                continue

            # Prepare training and validation data
            data = {
                "X_train": train_df[feature_set].values,
                "y_train": train_df[target].values,
                "X_val": val_df[feature_set].values,
                "y_val": val_df[target].values,
            }

            # Configuration for the current model
            config = {
                "output_path": output_path,
                "unit_id": name,
                "feature_set_name": feature_set_name,
            }

            # Initialize and train the model
            model = ModelClass(input_size=len(feature_set)) if model_type == "LSTM_PYTORCH" else ModelClass()
            model.train_and_save(data, config)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train models for disease case prediction.")

    # Add arguments
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset file (Parquet format).")
    parser.add_argument("--output_path", required=True, help="Path to save trained models and outputs.")
    parser.add_argument("--train_cutoff", required=True, help="Date cutoff for training data (YYYY-MM-DD).")
    parser.add_argument("--val_cutoff", required=True, help="Date cutoff for validation data (YYYY-MM-DD).")
    parser.add_argument("--model_type", required=True, choices=["LSTM_PYTORCH", "XGBoost"], help="Type of model to train.")
    parser.add_argument("--id_unidade", required=False, help="ID_UNIDADE to filter data for training (optional).", default=None)

    # Parse arguments
    args = parser.parse_args()

    # Call the train function with parsed arguments
    train(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        train_cutoff=args.train_cutoff,
        val_cutoff=args.val_cutoff,
        model_type=args.model_type,
        id_unidade=args.id_unidade,
    )

    # train(
    #     dataset_path='data\processed\sinan\sinan.parquet',
    #     output_path='data\processed',
    #     train_cutoff='2021-12-31',
    #     val_cutoff='2022-12-31',
    #     model_type='LSTM_PYTORCH',
    #     id_unidade='2283395',
    # )    
