import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def evaluate_rf_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)

    def metrics(label, y_true, y_pred):
        y_pred = np.maximum(y_pred, 0)  # garantir que não haja negativos
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        non_zero_mask = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100 \
            if np.any(non_zero_mask) else float('nan')

        print(f"\n📊 [{label}] - {name}")
        print(f"  MSE : {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE : {mae:.4f}")
        print(f"  R²  : {r2:.4f}")
        print(f"  MAPE (ign. zeros): {mape:.2f}%")

    # Previsões
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Avaliação
    metrics("TREINO", y_train, y_pred_train)
    metrics("VALIDAÇÃO", y_val, y_pred_val)
    metrics("TESTE", y_test, y_pred_test)

    # 🔁 Recalcular previsões do modelo treinado no conjunto de teste
    treino_teste = np.maximum(model.predict(X_test), 0)
    y_true = y_test

    # 🔍 Erros e métricas detalhadas
    erro_abs = np.abs(y_true - treino_teste)
    erro = y_true - treino_teste
    rmse_total = np.sqrt(mean_squared_error(y_true, treino_teste))

    # 📊 Salvar resultado em CSV
    y_true_data = pd.DataFrame(y_true, columns=["Cases"])
    simulado_data = pd.DataFrame(treino_teste, columns=["Treino"])
    erro_abs_data = pd.DataFrame(erro_abs, columns=["Erro Absoluto"])
    erro_data = pd.DataFrame(erro, columns=["Erro"])

    result = pd.concat([y_true_data, simulado_data, erro_abs_data, erro_data], axis=1)
    result.to_csv("result.csv", index=False)
    print("📁 Resultados salvos em 'result.csv'")

    # 🔬 Métricas finais no conjunto de teste
    erro_medio = np.mean(erro_abs)
    rmse = np.sqrt(mean_squared_error(y_true, treino_teste))
    r2 = r2_score(y_true, treino_teste)

    # Correlação de Pearson
    corr, _ = pearsonr(y_true, treino_teste)

    print(f"\n📈 Avaliação Final no Conjunto de Teste:")
    print(f"Erro absoluto médio: {erro_medio:.3f}")
    print(f"RMSE               : {rmse:.3f}")
    print(f"R²                 : {r2:.3f}")
    print(f"Correlação Pearson : {corr:.3f}")

    # 📊 Importância das variáveis (se disponível)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        print("\n🔎 Top 10 variáveis mais importantes:")
        for i in np.argsort(importances)[::-1][:10]:
            print(f"Feature {i}: Importance = {importances[i]:.4f}")

        return model, y_pred_test

# 🔁 Criação e avaliação do modelo Random Forest
# rf_model = RandomForestRegressor(
#     n_jobs=-1,
#     verbose=1,
#     criterion = 'poisson', 
#     n_estimators = 1500, 
#     max_features = 0.2, 
#     max_depth = 200,
#     bootstrap=True, 
#     random_state = 18)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=987,
    n_jobs=8,
    verbose=1
)

print("🔍 Treinando Random Forest com seus dados...")
filename = "data/datasets/RJ.pickle"
file = open(filename, 'rb')
(X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

# Reshape data if necessary
X_train = X_train.reshape(X_train.shape[0], -1) 
X_val = X_val.reshape(X_val.shape[0], -1)  
X_test = X_test.reshape(X_test.shape[0], -1)  

model_rf, y_pred_rf = evaluate_rf_model("Random Forest", rf_model,
                                        X_train, y_train, X_val, y_val, X_test, y_test)
                                        

joblib.dump(model_rf, "models/Random_Forest_RJ.pkl")