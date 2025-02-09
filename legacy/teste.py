import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from flask import send_from_directory

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Define the static folder path
STATIC_PATH = os.path.abspath("data/processed")

# Flask route to serve static files
@server.route("/static/<path:path>")
def serve_static_files(path):
    full_path = os.path.join(STATIC_PATH, path)
    if os.path.exists(full_path):
        return send_from_directory(STATIC_PATH, path)
    else:
        return "File Not Found", 404

# Function to collect training results
def collect_training_results(base_path, model_type):
    results = []
    for file in os.listdir(base_path):
        if file.endswith(".png"):
            parts = file.split("_")
            id_unidade = parts[-2]
            feature_set = "_".join(parts[-1:]).replace(".png", "")
            relative_path = os.path.relpath(os.path.join(base_path, file), STATIC_PATH)
            results.append({
                "ID_UNIDADE": id_unidade,
                "FEATURE_SET": feature_set,
                "MODEL_TYPE": model_type,
                "TRAIN_PLOT": f"/static/{relative_path.replace(os.sep, '/')}"
            })
    return results

# Function to collect prediction results
def collect_prediction_results(base_path, model_type):
    results = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.startswith("results_") and file.endswith(".txt"):
                    parts = file.replace("results_", "").replace(".txt", "").split("_")
                    id_unidade = parts[0]
                    feature_set = "_".join(parts[1:])
                    txt_file = os.path.join(folder_path, file)
                    plot_file = None

                    # Look for the corresponding plot file
                    for f in os.listdir(folder_path):
                        if f.endswith(".png"):
                            relative_plot_path = os.path.relpath(os.path.join(folder_path, f), STATIC_PATH)
                            plot_file = f"/static/{relative_plot_path.replace(os.sep, '/')}"

                    # Read metrics from the results file
                    if os.path.exists(txt_file):
                        with open(txt_file, "r") as f:
                            lines = f.readlines()
                            rmse = float(lines[0].split(":")[1].strip())
                            mae = float(lines[1].split(":")[1].strip())
                            r2 = float(lines[2].split(":")[1].strip())
                            results.append({
                                "ID_UNIDADE": id_unidade,
                                "FEATURE_SET": feature_set,
                                "MODEL_TYPE": model_type,
                                "RMSE": rmse,
                                "MAE": mae,
                                "R2": r2,
                                "PREDICTION_PLOT": plot_file
                            })
    return results

# Collect results
train_xgboost_path = os.path.join("data", "processed", "XGBOOST")
train_lstm_path = os.path.join("data", "processed", "LSTM_PYTORCH")
pred_xgboost_path = os.path.join("data", "processed", "XGBOOST_RESULTS")
pred_lstm_path = os.path.join("data", "processed", "LSTM_PYTORCH_RESULTS")

train_results = []
train_results.extend(collect_training_results(train_xgboost_path, "XGBoost"))
train_results.extend(collect_training_results(train_lstm_path, "LSTM_PYTORCH"))

prediction_results = []
prediction_results.extend(collect_prediction_results(pred_xgboost_path, "XGBoost"))
prediction_results.extend(collect_prediction_results(pred_lstm_path, "LSTM_PYTORCH"))

train_df = pd.DataFrame(train_results)
pred_df = pd.DataFrame(prediction_results)

# Merge results for visualization
merged_df = pd.merge(pred_df, train_df, on=["ID_UNIDADE", "FEATURE_SET", "MODEL_TYPE"], how="left")

# Layout for the dashboard
app.layout = html.Div([
    html.H1("Training and Prediction Results Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Model Type:"),
        dcc.Dropdown(
            id="model-type-dropdown",
            options=[
                {"label": "XGBoost", "value": "XGBoost"},
                {"label": "LSTM_PYTORCH", "value": "LSTM_PYTORCH"}
            ],
            value="XGBoost",
            style={"width": "50%"}
        ),
        html.Label("Select ID_UNIDADE:"),
        dcc.Dropdown(
            id="id-unidade-dropdown",
            style={"width": "50%"}
        ),
        html.Label("Select Feature Set:"),
        dcc.Dropdown(
            id="feature-set-dropdown",
            style={"width": "50%"}
        ),
    ]),

    html.Div(id="metrics-output", style={"marginTop": "20px"}),

    html.Div([
        html.H2("Training Plot"),
        html.Img(id="train-plot", style={"maxWidth": "100%"}),
    ], style={"marginTop": "20px"}),

    html.Div([
        html.H2("Prediction Plot"),
        html.Img(id="prediction-plot", style={"maxWidth": "100%"}),
    ], style={"marginTop": "20px"})
])

# Callbacks for interactivity
@app.callback(
    [Output("id-unidade-dropdown", "options"),
     Output("id-unidade-dropdown", "value")],
    Input("model-type-dropdown", "value")
)
def update_id_unidade_dropdown(model_type):
    options = [{"label": i, "value": i} for i in merged_df[merged_df["MODEL_TYPE"] == model_type]["ID_UNIDADE"].unique()]
    return options, options[0]["value"] if options else None


@app.callback(
    [Output("feature-set-dropdown", "options"),
     Output("feature-set-dropdown", "value")],
    [Input("model-type-dropdown", "value"),
     Input("id-unidade-dropdown", "value")]
)
def update_feature_set_dropdown(model_type, id_unidade):
    filtered_df = merged_df[(merged_df["MODEL_TYPE"] == model_type) & (merged_df["ID_UNIDADE"] == id_unidade)]
    options = [{"label": i, "value": i} for i in filtered_df["FEATURE_SET"].unique()]
    return options, options[0]["value"] if options else None


@app.callback(
    [Output("metrics-output", "children"),
     Output("train-plot", "src"),
     Output("prediction-plot", "src")],
    [Input("model-type-dropdown", "value"),
     Input("id-unidade-dropdown", "value"),
     Input("feature-set-dropdown", "value")]
)
def update_visuals(model_type, id_unidade, feature_set):
    filtered_df = merged_df[
        (merged_df["MODEL_TYPE"] == model_type) &
        (merged_df["ID_UNIDADE"] == id_unidade) &
        (merged_df["FEATURE_SET"] == feature_set)
    ]
    if not filtered_df.empty:
        row = filtered_df.iloc[0]
        metrics = [
            html.P(f"RMSE: {row['RMSE']:.4f}"),
            html.P(f"MAE: {row['MAE']:.4f}"),
            html.P(f"R²: {row['R2']:.4f}")
        ]
        return metrics, row["TRAIN_PLOT"], row["PREDICTION_PLOT"]
    return ["Metrics not available for the selected combination."], None, None

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
