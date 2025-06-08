import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# 📌 Load dataset
filename = "data/datasets/FULL.pickle"
file = open(filename, 'rb')
(X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

# Reshape data if necessary
X_train = X_train.reshape(X_train.shape[0], -1) 
X_val = X_val.reshape(X_val.shape[0], -1)  
X_test = X_test.reshape(X_test.shape[0], -1)  

# 📊 📌 Check data distribution to detect potential train/test bias
plt.figure(figsize=(8, 6))
plt.hist(y_train, bins=30, alpha=0.5, label="Train")
plt.hist(y_test, bins=30, alpha=0.5, label="Test")
plt.legend()
plt.title("Target Variable Distribution (Train vs Test)")
plt.show()

# 📌 Check for Data Leakage
overlap_features = set(X_train.flatten()) & set(y_train.flatten())
if overlap_features:
    print(f"Possible Data Leakage Detected! {len(overlap_features)} overlapping values found.")

# 📌 Option: Apply Log Transformation to Target Variable (to reduce large value impact)
use_log_transform = True  # Set to False if you don’t want to use it
if use_log_transform:
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

# 📌 Train XGBoost Poisson Regression Model
model = XGBRegressor(
    objective="count:poisson",
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

# 📊 📌 Plot training progression
results = model.evals_result()
plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['poisson-nloglik'], label='Train')
plt.plot(results['validation_1']['poisson-nloglik'], label='Validation')
plt.xlabel("Iterations")
plt.ylabel("Negative Loglikelihood")
plt.title("XGBoost Training Progression")
plt.legend()
plt.show()

# 📌 Make predictions
y_pred = model.predict(X_test)

# Reverse Log Transformation if applied
if use_log_transform:
    y_pred = np.expm1(y_pred)

# 📊 📌 Plot actual vs. predicted values
predictions_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
plt.figure(figsize=(8, 6))
sns.scatterplot(x=predictions_df["Actual"], y=predictions_df["Predicted"], alpha=0.5)
plt.plot([predictions_df["Actual"].min(), predictions_df["Actual"].max()],
         [predictions_df["Actual"].min(), predictions_df["Actual"].max()], color="red", linestyle="--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values - XGBoost Poisson Regression")
plt.show()

# 📌 Calculate model evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 📌 Fix MAPE Calculation to Ignore Zero Values
non_zero_mask = y_test != 0
if np.any(non_zero_mask):  # Ensure we don't divide by zero
    mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
else:
    mape = np.nan  # Return NaN if all values are zero to avoid misleading results

print(f"Recomputed MSE: {mse:.4f}")
print(f"Recomputed RMSE: {rmse:.4f}")
print(f"Recomputed MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE (Ignoring Zeros): {mape:.2f}%")

# 📌 Compare train vs. test errors to check for overfitting
train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, y_pred)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# 📌 Break down MAE for small vs. large values
low_value_mask = y_test < 10
high_value_mask = y_test >= 10

# Ensure we have values in both categories before computing MAE
if np.any(low_value_mask):
    mae_low = mean_absolute_error(y_test[low_value_mask], y_pred[low_value_mask])
    print(f"MAE for low values (<10): {mae_low:.4f}")
else:
    print("No values found in the <10 range, skipping MAE_low calculation.")

if np.any(high_value_mask):
    mae_high = mean_absolute_error(y_test[high_value_mask], y_pred[high_value_mask])
    print(f"MAE for high values (>=10): {mae_high:.4f}")
else:
    print("No values found in the >=10 range, skipping MAE_high calculation.")

# 📊 📌 Error distribution and residuals
errors = y_test - y_pred

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 📌 1. Histogram of residuals (error distribution)
sns.histplot(errors, bins=30, kde=True, ax=axs[0])
axs[0].axvline(x=0, color='red', linestyle='--')
axs[0].set_xlabel("Prediction Error (Actual - Predicted)")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Distribution of Prediction Errors")

# 📌 2. Residuals vs. Actual Values
sns.scatterplot(x=y_test, y=errors, alpha=0.5, ax=axs[1])
axs[1].axhline(y=0, color='red', linestyle='--')
axs[1].set_xlabel("Actual Values")
axs[1].set_ylabel("Residuals (Actual - Predicted)")
axs[1].set_title("Residuals vs Actual Values")

plt.tight_layout()
plt.show()

# 📌 Q-Q Plot for Residuals Normality Check
plt.figure(figsize=(6, 6))
stats.probplot(errors, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

# 📌 Alternative Model: Train Quantile Regression for More Robust Predictions
use_quantile_regression = True  # Set to False to disable
if use_quantile_regression:
    quantile_model = GradientBoostingRegressor(loss="quantile", alpha=0.5)
    quantile_model.fit(X_train, y_train)
    y_pred_quantile = quantile_model.predict(X_test)
    
    # Reverse Log Transformation if applied
    if use_log_transform:
        y_pred_quantile = np.expm1(y_pred_quantile)

    quantile_mae = mean_absolute_error(y_test, y_pred_quantile)
    print(f"Quantile Regression MAE: {quantile_mae:.4f}")
