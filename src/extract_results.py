import os
import re
import pandas as pd

def extract_metrics_from_txt(text):
    sections = ["TREINO", "VALIDAÇÃO", "TESTE"]
    results = []

    for section in sections:
        pattern = rf"\[({section})\](.*?)\n\n"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            metrics_text = match.group(2)
            metrics = {
                "Conjunto": section.capitalize(),
                "MSE": float(re.search(r"MSE\s*:\s*([\d.]+)", metrics_text).group(1)),
                "RMSE": float(re.search(r"RMSE\s*:\s*([\d.]+)", metrics_text).group(1)),
                "MAE": float(re.search(r"MAE\s*:\s*([\d.]+)", metrics_text).group(1)),
                "R2": float(re.search(r"R(?:2|²)\s*:\s*(-?[\d.]+)", metrics_text).group(1)),
                "MAPE (%)": float(re.search(r"MAPE.*?:\s*([\d.]+)", metrics_text).group(1))
            }
            results.append(metrics)
    
    return results

# Caminho principal
base_path = 'test_full'

# Lista de resultados
all_results = []

# Percorrer subpastas
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".txt"):
            txt_path = os.path.join(root, file)
            model_name = os.path.basename(root)  # nome da subpasta
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            metrics = extract_metrics_from_txt(text)
            for m in metrics:
                m["Modelo"] = model_name
                all_results.append(m)

# Criar DataFrame consolidado
df = pd.DataFrame(all_results)

# Reorganizar colunas
df = df[["Modelo", "Conjunto", "MSE", "RMSE", "MAE", "R2", "MAPE (%)"]]

# Salvar como CSV
output_csv = os.path.join(base_path, "metricas_consolidadas.csv")
df.to_csv(output_csv, index=False)

print(f"✔️ Métricas extraídas de {len(df['Modelo'].unique())} modelos e salvas em: {output_csv}")
