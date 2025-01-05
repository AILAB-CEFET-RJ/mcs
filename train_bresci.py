import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import argparse


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

    def train_and_save(self, X_train, y_train, X_val, y_val, output_path, unit_id, feature_set_name):
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
            torch.nn.Linear(hidden_size, 1)
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

    def train_and_save(self, X_train, y_train, X_val, y_val, output_path, unit_id, feature_set_name):
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


# XGBoost Model Class
class XGBoostModel(BaseModel):
    def __init__(self, name="XGBoost_Model"):
        super().__init__(name)
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=128, learning_rate=0.1)

    def save_model(self, output_path, unit_id, feature_set_name):
        model_path = os.path.join(output_path, f"{self.name}_{unit_id}_{feature_set_name}.json")
        os.makedirs(output_path, exist_ok=True)
        self.model.save_model(model_path)
        logging.info(f"Model saved to {model_path}")

    def train_and_save(self, X_train, y_train, X_val, y_val, output_path, unit_id, feature_set_name):
        X_train, X_val = self.prepare_data(X_train, X_val)

        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.model.set_params(early_stopping_rounds=10)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

        self.save_model(output_path, unit_id, feature_set_name)


# Training Function
def train(dataset_path, output_path, train_cutoff, val_cutoff, model_type, id_unidade=None):
    df = pd.read_parquet(dataset_path)
    if id_unidade:
        df = df[df["ID_UNIDADE"] == id_unidade]

    # Define feature groups
    features_cases = ['CASES', 'CASES_MM_14', 'CASES_MM_21', 'CASES_ACC_14', 'CASES_ACC_21']
    features_inmet = ['TEMP_MIN', 'TEMP_MAX', 'TEMP_MEAN', 'HUMIDITY', 'RAIN']
    features_sat = ['TEMP_AVG_SAT', 'RAIN_SAT']

    all_features = features_cases + features_inmet + features_sat
    inmet_and_cases = features_cases + features_inmet
    sat_and_cases = features_cases + features_sat

    training_features = {
        "all_features": all_features,
        "inmet_and_cases": inmet_and_cases,
        "sat_and_cases": sat_and_cases,
        "cases": features_cases,
    }

    target = "CASES"
    df = df.dropna()

    grouped = df.groupby("ID_UNIDADE")

    if model_type == "LSTM_PYTORCH":
        ModelClass = LSTM_PyTorchModel
    elif model_type == "XGBoost":
        ModelClass = XGBoostModel
    else:
        raise ValueError("Invalid model type!")

    for name, group in grouped:
        group = group.sort_values(by="DT_NOTIFIC")
        train_df = group[group["DT_NOTIFIC"] <= train_cutoff]
        val_df = group[(group["DT_NOTIFIC"] > train_cutoff) & (group["DT_NOTIFIC"] <= val_cutoff)]

        for feature_set_name, feature_set in training_features.items():
            X_train, y_train = train_df[feature_set].values, train_df[target].values
            X_val, y_val = val_df[feature_set].values, val_df[target].values

            model = ModelClass(input_size=len(feature_set)) if model_type == "LSTM_PYTORCH" else ModelClass()
            model.train_and_save(X_train, y_train, X_val, y_val, output_path, name, feature_set_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train(
        dataset_path="data/sinan/sinan.parquet",
        output_path="data/output",
        train_cutoff="2021-12-31",
        val_cutoff="2022-12-31",
        model_type="LSTM_PYTORCH",
        id_unidade=None,
    )
