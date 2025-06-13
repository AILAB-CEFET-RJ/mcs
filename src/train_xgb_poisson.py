import argparse
import os
import time
import joblib
import numpy as np


from data.data_utils import load_data
from evaluation.eval_utils import get_training_metrics, save_all_metrics, save_feature_importance
from evaluation.plots import plot_learning_curve, plot_prediction_distribution
from models.models import get_xgb_poisson


def train_and_evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test, outdir, feature_dictionary):   
    
    print(f"Ajustando modelo...")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    print(f"Prevendo...")
    if hasattr(model, 'best_iteration'):
        y_pred_test = model.predict(X_test, iteration_range=(0, model.best_iteration + 1))
    else:
        y_pred_test = model.predict(X_test)
        
    y_pred_train = np.round(np.maximum(model.predict(X_train), 0)).astype(int)
    y_pred_val = np.round(np.maximum(model.predict(X_val), 0)).astype(int)
    y_pred_test = np.round(np.maximum(model.predict(X_test), 0)).astype(int)

    print(f"Carregando métricas de treino...")
    metrics_dict = {
        "Treino": get_training_metrics(y_train, y_pred_train),
        "Validação": get_training_metrics(y_val, y_pred_val),
        "Teste": get_training_metrics(y_test, y_pred_test)
    }

    print(f"Salvando métricas de treino...")
    save_all_metrics(metrics_dict, outdir)
    
    print(f"Plotando curva de aprendizado...")
    plot_learning_curve(model, outdir, name)    
    
    print(f"Plotando distribuição de predições...")
    plot_prediction_distribution(y_pred_test, name, outdir)    

    print(f"Salvando importância de features...")
    if hasattr(model, "feature_importances_"):
        save_feature_importance(name, model, X_train, outdir, feature_dictionary)

    return model, y_pred_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de modelo para previsão de dengue")
    parser.add_argument("--dataset", type=str, required=True, help="Caminho para o arquivo .pickle com os dados")
    parser.add_argument("--outdir", type=str, required=True, help="Diretório para salvar resultados")
    parser.add_argument("--seed", type=int, required=True, help="Seed")    
    parser.add_argument("--dict", type=str, required=True, help="Feature dictionary")    

    args = parser.parse_args()
    start_time = time.time() 
    
    seed = 987
    if args.seed:
        seed = args.seed
    model = get_xgb_poisson(seed)
    
    print(f"Carregando dados...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    model, y_pred = train_and_evaluate('XGBOOST - Poisson', model,
                                   X_train, y_train, X_val, y_val, X_test, y_test,
                                   outdir=args.outdir, feature_dictionary=args.dict)

    model_path = os.path.join(args.outdir, f"model_xgboost_poisson.pkl")
    joblib.dump(model, model_path)
    print(f"📂 Modelo salvo em: {model_path}")
    
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    
    print(f"Script duration: {duration:.2f} minutes")        
