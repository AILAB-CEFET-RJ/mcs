import argparse
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_poisson_deviance, mean_tweedie_deviance
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_model(model_name, seed):
    model_name = model_name.lower()
    if model_name == "randomforest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=seed,
            n_jobs=8,
            verbose=1
        )
    elif model_name == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=seed,
            n_jobs=8,
            verbosity=1,
            early_stopping_rounds=10,
            objective="reg:squarederror"
        )
    elif model_name == "xgboost_poisson":
        return XGBRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=seed,
            n_jobs=8,
            verbosity=1,
            objective="count:poisson",
            early_stopping_rounds=10,
            eval_metric="poisson-nloglik"
        )
    elif model_name == "xgboost_tweedie":
        return XGBRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=seed,
            n_jobs=8,
            verbosity=1,
            objective="reg:tweedie",
            early_stopping_rounds=10,
            tweedie_variance_power=1.1
        )
    else:
        raise ValueError(f"\u274c Modelo não suportado: {model_name}")

def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    )
    
def plot_distribuicao(y_pred, nome_dataset, outdir):
    def faixa(valor):
        if valor == 0:
            return '0'
        elif 1 <= valor <= 3:
            return '1-3'
        elif 4 <= valor <= 6:
            return '4-6'
        else:
            return '7+'

    faixas = pd.Series([faixa(v) for v in y_pred])
    contagem = faixas.value_counts().reindex(['0', '1-3', '4-6', '7+'], fill_value=0)

    plt.figure(figsize=(8,6))
    contagem.plot(kind='bar', color='#4C72B0')

    plt.title(f'Distribuição das Predições por Faixa - {nome_dataset}')
    plt.xlabel('Faixa de Casos')
    plt.ylabel('Quantidade de Predições')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(contagem):
        plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{outdir}/{nome_dataset.lower().replace(" ", "_")}_distribuicao_predicoes.png')
    plt.close()
    
def plot_learning_curve(model, outdir, model_name):
    import matplotlib.pyplot as plt
    import os

    if hasattr(model, 'evals_result'):
        evals_result = model.evals_result()
        train_metric = None
        val_metric = None

        # Pegando o nome da métrica automaticamente
        metric_name = list(evals_result['validation_0'].keys())[0]
        train_metric = evals_result['validation_0'][metric_name]
        val_metric = evals_result['validation_1'][metric_name]

        plt.figure(figsize=(8, 6))
        plt.plot(train_metric, label='Treino')
        plt.plot(val_metric, label='Validação')
        plt.title(f"Learning Curve - {model_name}")
        plt.xlabel("Número de Árvores (Boosting Rounds)")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid()
        plt.tight_layout()

        save_path = os.path.join(outdir, f"learning_curve_{model_name.lower()}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"📈 Learning curve salva em: {save_path}")
    else:
        print("⚠️ Learning curve não disponível para este modelo.")   

def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test, outdir):   
    log_lines = []

    def metrics(label, y_true, y_pred):
        y_pred = np.maximum(y_pred, 0)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        non_zero_mask = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100 if np.any(non_zero_mask) else float('nan')
        try:
            poisson_dev = mean_poisson_deviance(y_true, y_pred)
            tweedie_dev = mean_tweedie_deviance(y_true, y_pred, power=1.1)
        except:
            poisson_dev = tweedie_dev = float('nan')
        smape_val = smape(y_true, y_pred)

        log_lines.append(f"\n📊 [{label}] - {name}")
        log_lines.append(f"  MSE : {mse:.4f}")
        log_lines.append(f"  RMSE: {rmse:.4f}")
        log_lines.append(f"  MAE : {mae:.4f}")
        log_lines.append(f"  R²  : {r2:.4f}")
        log_lines.append(f"  MAPE (ign. zeros): {mape:.2f}%")
        log_lines.append(f"  SMAPE: {smape_val:.2f}%")
        log_lines.append(f"  Poisson Deviance : {poisson_dev:.4f}")
        log_lines.append(f"  Tweedie Deviance : {tweedie_dev:.4f}")

    if isinstance(model, XGBRegressor):
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    if hasattr(model, 'best_iteration'):
        y_pred_test = model.predict(X_test, iteration_range=(0, model.best_iteration + 1))
    else:
        y_pred_test = model.predict(X_test)
        
    y_pred_train = np.round(model.predict(X_train))
    y_pred_val = np.round(model.predict(X_val))
    y_pred_test = np.round(model.predict(X_test))

    metrics("TREINO", y_train, y_pred_train)
    metrics("VALIDAÇÃO", y_val, y_pred_val)
    metrics("TESTE", y_test, y_pred_test)
    if isinstance(model, XGBRegressor):
        plot_learning_curve(model, outdir, name)    

    treino_teste = np.round(np.maximum(model.predict(X_test), 0)).astype(int)
    y_true = y_test
    erro_abs = np.abs(y_true - treino_teste)
    erro = y_true - treino_teste

    result_df = pd.concat([
        pd.DataFrame(y_true, columns=["Cases"]),
        pd.DataFrame(treino_teste, columns=["Predito"]),
        pd.DataFrame(erro_abs, columns=["Erro Absoluto"]),
        pd.DataFrame(erro, columns=["Erro"])
    ], axis=1)

    result_csv_path = os.path.join(outdir, f"predictions_{name.lower()}.csv")
    result_df.to_csv(result_csv_path, index=False)
    
    plot_distribuicao(treino_teste, name, outdir)    

    erro_medio = np.mean(erro_abs)
    rmse = np.sqrt(mean_squared_error(y_true, treino_teste))
    r2 = r2_score(y_true, treino_teste)
    corr, _ = pearsonr(y_true, treino_teste)

    log_lines.append(f"\n📈 Avaliação Final no Conjunto de Teste:")
    log_lines.append(f"Erro absoluto médio: {erro_medio:.3f}")
    log_lines.append(f"RMSE               : {rmse:.3f}")
    log_lines.append(f"R²                 : {r2:.3f}")
    log_lines.append(f"Correlação Pearson : {corr:.3f}")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        # Carrega o nome original das features
        try:
            original_features = pd.read_csv("feature_dictionary.csv")["Feature"].tolist()
        except Exception as e:
            print(f"⚠️ Erro ao carregar feature_dictionary.csv: {e}")
            original_features = [f"f{i}" for i in range(X_train.shape[1] // window_size)]

        # Gera nomes expandidos considerando sliding window
        def get_sliding_feature_names(original_features, window_size):
            return [
                f"{f}_t-{i}" for i in reversed(range(window_size)) for f in original_features
            ]

        window_size = X_train.shape[1] // len(original_features)
        all_features = get_sliding_feature_names(original_features, window_size)

        if len(all_features) != len(importances):
            print(f"⚠️ Tamanho dos nomes de features ({len(all_features)}) difere de importances ({len(importances)}). Gerando nomes genéricos.")
            all_features = [f"Feature_{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame({
            "Feature": all_features,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        importance_df.to_csv(os.path.join(outdir, f"feature_importance_{name.lower()}.csv"), index=False)

        # Histograma das top 20
        import matplotlib.pyplot as plt
        top_n = 20
        plt.figure(figsize=(10, 6))
        importance_df.head(top_n).plot(kind='barh', x='Feature', y='Importance', legend=False)
        plt.title(f"Top {top_n} Features Mais Importantes - {name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"feature_importance_{name.lower()}.png"))
        plt.close()

    os.makedirs(outdir, exist_ok=True)
    result_txt_path = os.path.join(outdir, f"result_{name.lower()}.txt")
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return model, y_pred_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de modelo para previsão de dengue")
    parser.add_argument("--model", type=str, required=True,
                        choices=["randomforest", "xgboost", "xgboost_poisson", "xgboost_tweedie"],
                        help="Modelo a ser usado")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Caminho para o arquivo .pickle com os dados")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Diretório para salvar resultados")
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed")    

    args = parser.parse_args()

    print(f"🔍 Treinando modelo: {args.model}")
    seed = 987
    if args.seed:
        seed = args.seed
    model = get_model(args.model, seed)

    with open(args.dataset, "rb") as file:
        (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    model, y_pred = evaluate_model(args.model.capitalize(), model,
                                   X_train, y_train, X_val, y_val, X_test, y_test,
                                   outdir=args.outdir)

    model_path = os.path.join(args.outdir, f"model_{args.model.lower()}.pkl")
    joblib.dump(model, model_path)
    print(f"📂 Modelo salvo em: {model_path}")
