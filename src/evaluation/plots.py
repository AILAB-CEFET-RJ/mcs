import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import clone
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

def plot_prediction_distribution(y_pred, nome_dataset, outdir):
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
        plt.title(f"Boosting Loss Curve - {model_name}")
        plt.xlabel("Número de Árvores (Boosting Rounds)")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid()
        plt.tight_layout()

        save_path = os.path.join(outdir, f"boosting_loss_curve_{model_name.lower()}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"📈 Boosting Loss Curve salva em: {save_path}")
    else:
        print("⚠️ Boosting Loss Curve não disponível para este modelo.")
        
def plot_learning_curve_external(model, X_train, y_train, X_val, y_val, outdir, model_name):
    print("Gerando learning curve externa com validação fixa...")

    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs = (train_sizes * len(X_train)).astype(int)

    train_scores = []
    val_scores = []

    for size in train_sizes_abs:
        print(f"Treinando com {size} amostras...")
        # Amostragem sequencial (para time series é melhor não aleatorizar)
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]

        model_clone = clone(model)
        model_clone.fit(X_train_subset, y_train_subset)

        y_train_pred = model_clone.predict(X_train_subset)
        y_val_pred = model_clone.predict(X_val)

        train_mse = mean_squared_error(y_train_subset, np.round(np.maximum(y_train_pred, 0)).astype(int))
        val_mse = mean_squared_error(y_val, np.round(np.maximum(y_val_pred, 0)).astype(int))

        train_scores.append(train_mse)
        val_scores.append(val_mse)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes_abs, train_scores, 'o-', label='Treino')
    plt.plot(train_sizes_abs, val_scores, 'o-', label='Validação')

    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Número de amostras de treino")
    plt.ylabel("Erro Quadrático Médio (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(outdir, f"learning_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Learning curve salva em: {save_path}")       
   