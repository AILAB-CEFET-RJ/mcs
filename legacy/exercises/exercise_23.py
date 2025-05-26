import pickle
import argparse
import matplotlib.pyplot as plt
import os


def plot_pie(y, title, ax):
    zero_count = (y == 0).sum()
    non_zero_count = (y != 0).sum()
    total = len(y)

    ax.pie(
        [zero_count, non_zero_count],
        labels=['Zero', 'Não Zero'],
        autopct='%1.1f%%',
        colors=['#FF9999', '#99FF99'],
        startangle=90
    )
    ax.set_title(f"{title}\nTotal: {total} amostras", fontsize=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerar gráficos de pizza para proporção de zeros")
    parser.add_argument("--dataset", type=str, required=True, help="Caminho para o arquivo .pickle")
    parser.add_argument("--outdir", type=str, default=".", help="Diretório para salvar os gráficos")
    args = parser.parse_args()

    # 🔍 Nome do dataset (pegando do nome do arquivo)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    dataset_title = dataset_name.replace("_", " ")

    # 🔍 Carregar o dataset
    with open(args.dataset, "rb") as file:
        (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(file)

    # 🎨 Plotando os gráficos
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    plot_pie(y_train, "Treino", axs[0])
    plot_pie(y_val, "Validação", axs[1])
    plot_pie(y_test, "Teste", axs[2])

    plt.suptitle(f"{dataset_title} — Proporção de Zeros e Não Zeros", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # 💾 Salvar o gráfico
    output_path = os.path.join(args.outdir, f"proporcao_zeros_{dataset_name}.png")
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"✅ Gráfico salvo em {output_path}")
