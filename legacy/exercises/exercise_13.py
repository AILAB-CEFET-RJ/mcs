# Plot a X,Y position over time

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset

# Caminhos dos arquivos
path = 'data/processed/lst_ts/'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
files.sort(key=lambda x: datetime.strptime(x[:-3], "%Y%m%d%H%M"))

# Listas para armazenar os valores extraídos e as datas
values = []
dates = []

# Processar cada arquivo
for file_name in files:
    # Carregar o arquivo
    full_path = os.path.join(path, file_name)
    with Dataset(full_path) as data:
        # Identificar a chave dos dados e carregar os dados
        array_data = data['Band1'][:]
        
        # Extrair o valor na posição [9, 10], substituindo NaN por 0
        value = array_data[9, 10]
        if np.isnan(value):
            value = 0
        values.append(value)
        
        # Extrair a data do nome do arquivo e converter para formato legível
        date_str = file_name.split('/')[-1][:12]  # yyyymmddhhmn
        date = datetime.strptime(date_str, "%Y%m%d%H%M")
        dates.append(date)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(dates, values, marker='o')
plt.title("Valor na Posição [9, 10] ao Longo do Tempo")
plt.xlabel("Data e Hora")
plt.ylabel("Valor na Posição [9, 10] (NaN como 0)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.close()
