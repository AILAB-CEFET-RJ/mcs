#create a graph of cases by year

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar o arquivo Parquet
df = pd.read_parquet('data/processed/sinan/sinan.parquet')

# Criar uma nova coluna 'ANO' a partir da data, que está em 'DT_NOTIFIC'
df['ANO'] = pd.to_datetime(df['DT_NOTIFIC']).dt.year

# Agrupar os dados por 'ANO' e 'ID_UNIDADE', somando os casos
df_grouped = df.groupby(['ANO', 'ID_UNIDADE'])['CASES'].sum().unstack()

# Definir a largura das barras
bar_width = 0.2

# Criar o eixo X com os anos
anos = df_grouped.index
unidades = df_grouped.columns

# Criar o gráfico
fig, ax = plt.subplots(figsize=(12, 8))

# Gerar uma matriz de posições para as barras, para garantir que elas fiquem lado a lado
positions = np.arange(len(anos))

# Plotar barras para cada unidade
for i, unidade in enumerate(unidades):
    ax.bar(positions + i * bar_width, df_grouped[unidade], bar_width, label=f'Unidade {unidade}')

# Configurar os rótulos dos eixos
ax.set_xlabel('Ano')
ax.set_ylabel('Quantidade de Casos')
ax.set_title('Quantidade de Casos por Ano e ID_UNIDADE')

# Adicionar os rótulos dos anos no eixo X
ax.set_xticks(positions + bar_width * (len(unidades) - 1) / 2)
ax.set_xticklabels(anos)

# Adicionar a legenda
plt.legend(title='ID_UNIDADE')

# Ajustar layout
plt.tight_layout()

# Exibir o gráfico
plt.show()
