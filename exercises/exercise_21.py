# Train a xgboost model
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset from the specified Parquet file
df = pd.read_parquet('data/processed/sinan/sinan.parquet')

# Split data into features and target
features_cases = [
    'CASES_MM_14', 'CASES_MM_21', 'CASES_ACC_14', 'CASES_ACC_21'
]
features_inmet = [
    'IDEAL_TEMP_INMET', 'EXTREME_TEMP_INMET', 'SIGNIFICANT_RAIN_INMET', 'EXTREME_RAIN_INMET', 'TEMP_RANGE_INMET',
    'TEM_AVG_INMET_MM_7', 'TEM_AVG_INMET_MM_14', 'TEM_AVG_INMET_MM_21',
    'CHUVA_INMET_MM_7', 'CHUVA_INMET_MM_14', 'CHUVA_INMET_MM_21',
    'TEMP_RANGE_INMET_MM_7', 'TEMP_RANGE_INMET_MM_14', 'TEMP_RANGE_INMET_MM_21',
    'TEM_AVG_INMET_ACC_7', 'TEM_AVG_INMET_ACC_14', 'TEM_AVG_INMET_ACC_21',
    'CHUVA_INMET_ACC_7', 'CHUVA_INMET_ACC_14', 'CHUVA_INMET_ACC_21'
]
features_sat = [
    'IDEAL_TEMP_SAT', 'EXTREME_TEMP_SAT', 'SIGNIFICANT_RAIN_SAT', 'EXTREME_RAIN_SAT', 'TEMP_RANGE_SAT',
    'TEM_AVG_SAT_MM_7', 'TEM_AVG_SAT_MM_14', 'TEM_AVG_SAT_MM_21',
    'CHUVA_SAT_MM_7', 'CHUVA_SAT_MM_14', 'CHUVA_SAT_MM_21',
    'TEMP_RANGE_SAT_MM_7', 'TEMP_RANGE_SAT_MM_14', 'TEMP_RANGE_SAT_MM_21',
    'TEM_AVG_SAT_ACC_7', 'TEM_AVG_SAT_ACC_14', 'TEM_AVG_SAT_ACC_21',
    'CHUVA_SAT_ACC_7', 'CHUVA_SAT_ACC_14', 'CHUVA_SAT_ACC_21'
]

# Define features and target
X = df[features_cases + features_inmet + features_sat]
y = df['CASES']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and configure the XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression task
    n_estimators=100,              # Number of boosting rounds
    max_depth=5,                   # Maximum depth of each tree
    learning_rate=0.1,             # Step size shrinkage
    subsample=0.8,                 # Fraction of samples to use per tree
    colsample_bytree=0.8,          # Fraction of features to use per tree
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"RMSE: {rmse:.2f}")

# Plot 1: Actual vs. Predicted Cases
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel('Actual Cases')
plt.ylabel('Predicted Cases')
plt.title('Actual vs. Predicted Dengue Cases')
plt.show()

# Plot 2: Feature Importance Plot
plt.figure(figsize=(12, 8))
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='weight')
plt.title("Feature Importance")
plt.show()
