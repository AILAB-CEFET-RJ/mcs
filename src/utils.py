import numpy as np
import matplotlib.pyplot as plt
import os

def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    )

def plot_learning_curve(model, outdir, model_name):
    if hasattr(model, 'evals_result'):
        evals_result = model.evals_result()
        metric_name = list(evals_result['validation_0'].keys())[0]
        train_metric = evals_result['validation_0'][metric_name]
        val_metric = evals_result['validation_1'][metric_name]

        plt.figure(figsize=(8, 6))
        plt.plot(train_metric, label='Treino')
        plt.plot(val_metric, label='Validação')
        plt.title(f"Learning Curve - {model_name}")
        plt.xlabel("Boosting Rounds")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        save_path = os.path.join(outdir, f"learning_curve_{model_name.lower()}.png")
        plt.savefig(save_path)
        plt.close()