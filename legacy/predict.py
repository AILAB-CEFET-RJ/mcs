import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import torch
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse


def load_model(model_path, model_type, input_size=None):
    """
    Load the saved model weights.
    """
    if model_type == "LSTM_PYTORCH":
        model = torch.nn.Sequential(
            torch.nn.LSTM(input_size, 50, batch_first=True),
            torch.nn.Linear(50, 1),
            torch.nn.ReLU()  # Ensures non-negative outputs
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model, device
    elif model_type == "XGBoost":
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_feature_set(feature_set_name):
    """
    Returns the column names for the specified feature set.
    """
    features_cases = ['CASES', 'CASES_MM_14', 'CASES_MM_21', 'CASES_ACC_14', 'CASES_ACC_21']
    windows = [7, 14, 21]
    features_era5 = ['TEM_MIN_ERA5', 'TEM_AVG_ERA5', 'TEM_MAX_ERA5',
                     'IDEAL_TEMP_ERA5', 'EXTREME_TEMP_ERA5', 'TEMP_RANGE_ERA5'] + [
                    f'TEM_AVG_ERA5_MM_{window}' for window in windows] + [
                    f'TEMP_RANGE_ERA5_MM_{window}' for window in windows] + [
                    f'TEM_AVG_ERA5_ACC_{window}' for window in windows]

    feature_sets = {
        "cases": features_cases,
        "all_features": features_cases + features_era5,
    }

    if feature_set_name not in feature_sets:
        raise ValueError(f"Feature set {feature_set_name} is not defined.")
    return feature_sets[feature_set_name]


def load_and_filter_data(dataset_path, id_unidade, date_start, date_end):
    """
    Load the dataset and filter by ID_UNIDADE and date range.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    df["ID_UNIDADE"] = df["ID_UNIDADE"].astype(str)
    df = df[df["ID_UNIDADE"] == str(id_unidade)]
    df = df[(df["DT_NOTIFIC"] >= date_start) & (df["DT_NOTIFIC"] <= date_end)]

    if df.empty:
        raise ValueError(f"No data available for ID_UNIDADE: {id_unidade} in the date range {date_start} to {date_end}")

    return df.dropna()


def save_results(output_path, id_unidade, feature_set_name, dates, y_actual, y_pred, rmse, mae, r2):
    """
    Save prediction results and metrics.
    """
    os.makedirs(output_path, exist_ok=True)

    # Save metrics
    results_path = os.path.join(output_path, f"results_{id_unidade}_{feature_set_name}.txt")
    with open(results_path, "w") as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R²: {r2:.4f}\n")
    logging.info(f"Results saved to {results_path}")

    # Save predictions
    predictions_df = pd.DataFrame({
        "Date": dates,
        "Actual": y_actual,
        "Predicted": y_pred
    })
    predictions_path = os.path.join(output_path, f"predictions_{id_unidade}_{feature_set_name}.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logging.info(f"Predictions saved to {predictions_path}")


def plot_predictions(dates, y_actual, y_pred, id_unidade, feature_set_name, output_path):
    """
    Generate and save a plot comparing actual vs predicted values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_actual, label="Actual Cases", marker="o", linestyle="-")
    plt.plot(dates, y_pred, label="Predicted Cases", marker="x", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.title(f"Predictions for ID_UNIDADE: {id_unidade}, Feature Set: {feature_set_name}")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_path, f"plot_{id_unidade}_{feature_set_name}.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Plot saved to {plot_path}")


def predict_and_evaluate(dataset_path, model_path, id_unidade, feature_set_name, date_start, date_end, model_type, output_path):
    """
    Predict and evaluate disease cases using a pre-trained model.
    """
    logging.info("Starting prediction and evaluation...")

    # Load and filter data
    df = load_and_filter_data(dataset_path, id_unidade, date_start, date_end)
    features = get_feature_set(feature_set_name)
    X = df[features].values
    y_actual = df["CASES"].values
    dates = pd.to_datetime(df["DT_NOTIFIC"])

    # Load model
    logging.info(f"Loading model from {model_path}")
    if model_type == "LSTM_PYTORCH":
        model, device = load_model(model_path, model_type, input_size=len(features))
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            y_pred = model[0](X)[0]
            y_pred = model[1](y_pred).cpu().numpy().flatten()
    elif model_type == "XGBoost":
        model = load_model(model_path, model_type)
        y_pred = model.predict(X)
    else:
        raise ValueError("Unsupported model type")

    # Post-process predictions
    y_pred = np.clip(np.round(y_pred), 0, None)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    logging.info(f"Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Save results and plot
    save_results(output_path, id_unidade, feature_set_name, dates, y_actual, y_pred, rmse, mae, r2)
    plot_predictions(dates, y_actual, y_pred, id_unidade, feature_set_name, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Predict and evaluate disease cases.")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset file")
    parser.add_argument("--model_path", required=True, help="Path to the saved model file")
    parser.add_argument("--id_unidade", required=True, help="Unit ID for prediction")
    parser.add_argument("--feature_set_name", required=True, choices=["cases", "all_features"], help="Feature set name")
    parser.add_argument("--date_start", required=True, help="Start date for prediction (YYYY-MM-DD)")
    parser.add_argument("--date_end", required=True, help="End date for prediction (YYYY-MM-DD)")
    parser.add_argument("--model_type", required=True, choices=["LSTM_PYTORCH", "XGBoost"], help="Model type")
    parser.add_argument("--output_path", required=True, help="Directory to save results")

    args = parser.parse_args()

    predict_and_evaluate(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        id_unidade=args.id_unidade,
        feature_set_name=args.feature_set_name,
        date_start=args.date_start,
        date_end=args.date_end,
        model_type=args.model_type,
        output_path=args.output_path,
    )