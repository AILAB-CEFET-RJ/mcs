import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminho principal
base_path = "test"

# Percorrer todas as subpastas
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".csv") and "prediction" in file.lower():
            csv_path = os.path.join(root, file)
            try:
                df = pd.read_csv(csv_path)

                # Verificar se colunas esperadas existem
                if 'Cases' in df.columns and 'Predito' in df.columns:
                    df['residual'] = df['Cases'] - df['Predito']

                    # Criar gráfico
                    plt.figure(figsize=(8, 5))
                    plt.scatter(df['Cases'], df['residual'], alpha=0.6, s=10)
                    plt.axhline(0, color='red', linestyle='--')
                    plt.xlabel("Casos Reais")
                    plt.ylabel("Resíduo (Real - Predito)")
                    plt.title(f"Resíduos - {os.path.basename(root)}")
                    plt.grid(True)
                    plt.tight_layout()

                    # Salvar imagem na mesma pasta
                    output_path = os.path.join(root, "residuos.png")
                    plt.savefig(output_path)
                    plt.close()

                    print(f"✔️ Imagem gerada para: {csv_path}")
                else:
                    print(f"⚠️ CSV inválido (faltam colunas 'Cases' e 'Predito'): {csv_path}")

            except Exception as e:
                print(f"❌ Erro ao processar {csv_path}: {e}")
