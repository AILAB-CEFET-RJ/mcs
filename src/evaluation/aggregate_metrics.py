# === aggregate_metrics.py ===

import os
import pandas as pd

def parse_model_folder_name(folder_name):
    """
    Exemplo de pasta: RJ_8_xgb_poisson_rj_weekly_casesonly
    """
    parts = folder_name.split('_')
    dataset = parts[0]
    seed = parts[1]
    model_type = parts[2] + (f"_{parts[3]}" if len(parts) >= 4 else "")
    variant = "_".join(parts[4:]) if len(parts) > 4 else ""
    
    return dataset, seed, model_type, variant

def collect_all_metrics(models_dir="models"):
    all_records = []
    
    for root, dirs, files in os.walk(models_dir):
        if "metrics.csv" in files:
            metrics_path = os.path.join(root, "metrics.csv")
            try:
                df = pd.read_csv(metrics_path)
                
                folder_name = os.path.basename(root)
                dataset, seed, model_type, variant = parse_model_folder_name(folder_name)
                
                df["Dataset"] = dataset
                df["Seed"] = seed
                df["Model"] = model_type
                df["Variant"] = variant
                
                all_records.append(df)
            except Exception as e:
                print(f"⚠️ Erro ao processar {metrics_path}: {e}")

    if all_records:
        final_df = pd.concat(all_records, ignore_index=True)
        # reorganizar as colunas
        cols = ["Dataset", "Variant", "Model", "Seed", "Conjunto"] + \
               [col for col in final_df.columns if col not in ["Dataset", "Variant", "Model", "Seed", "Conjunto"]]
        final_df = final_df[cols]
        return final_df
    else:
        print("Nenhum metrics.csv encontrado.")
        return pd.DataFrame()

def main():
    models_path = "models"  # caminho principal dos modelos
    output_path = "consolidated_metrics.csv"

    df_resultados = collect_all_metrics(models_path)
    df_resultados.to_csv(output_path, index=False)

    print(f"✅ Consolidação concluída: {output_path}")
    print(df_resultados.head())

if __name__ == "__main__":
    main()
