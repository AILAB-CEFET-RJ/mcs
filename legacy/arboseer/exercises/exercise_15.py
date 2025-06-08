# Unify INMET parquets
import pandas as pd
import os

def unificar_parquets(diretorio_input, arquivo_output):
    # Colunas que você quer manter
    colunas_desejadas = ['CD_ESTACAO', 'DT_MEDICAO', 'HR_MEDICAO', 'TEM_MIN', 'TEM_MAX', 'TEM_INS', 'VL_LATITUDE', 'VL_LONGITUDE']
    
    # Lista para armazenar os dataframes carregados de cada arquivo parquet
    dfs = []
    
    # Percorre todos os arquivos no diretório especificado
    for filename in os.listdir(diretorio_input):
        if filename.endswith('.parquet'):
            # Caminho completo do arquivo
            file_path = os.path.join(diretorio_input, filename)
            # Carrega o arquivo parquet para um dataframe
            df = pd.read_parquet(file_path)
            # Filtra as colunas desejadas
            df = df[colunas_desejadas]
            # Adiciona à lista
            dfs.append(df)
    
    # Concatena todos os dataframes em um único dataframe
    if dfs:
        df_concatenado = pd.concat(dfs, ignore_index=True)
        # Salva o dataframe concatenado como um arquivo parquet
        df_concatenado.to_parquet(arquivo_output)
        df_concatenado.to_csv('teste.csv')
        print(f"Arquivo '{arquivo_output}' criado com sucesso.")
    else:
        print("Não foram encontrados arquivos parquet no diretório.")

# Chama a função com o caminho da pasta onde estão os arquivos e o caminho/nome do arquivo de saída
unificar_parquets('data/raw/inmet_ts', 'data/raw/inmet_ts/concat.parquet')
