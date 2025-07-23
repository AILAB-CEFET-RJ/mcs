import pandas as pd
import matplotlib.pyplot as plt

# Carregar dados RJ e Natal
df_rj = pd.read_parquet('data/processed/sinan/DENG33.parquet')
df_natal = pd.read_parquet('data/processed/sinan/DENG240810.parquet')

# Limpar registros sem informação
df_rj = df_rj.dropna(subset=['CASES'])
df_natal = df_natal.dropna(subset=['CASES'])

# Contar zeros e não-zeros para RJ
total_rj = len(df_rj)
zeros_rj = (df_rj['CASES'] == 0).sum()
non_zeros_rj = total_rj - zeros_rj

# Contar zeros e não-zeros para Natal
total_natal = len(df_natal)
zeros_natal = (df_natal['CASES'] == 0).sum()
non_zeros_natal = total_natal - zeros_natal

# Dados para as pizzas
labels = ['Zeros', 'Não-Zeros']
colors = ['#4e79a7', '#f28e2c']
explode = (0.05, 0)

# Criar figura
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico RJ
axes[0].pie(
    [zeros_rj, non_zeros_rj],
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    shadow=True,
    startangle=90
)
axes[0].text(0, -1.3, 'Rio de Janeiro', ha='center', fontsize=14)

# Gráfico Natal
axes[1].pie(
    [zeros_natal, non_zeros_natal],
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    shadow=True,
    startangle=90
)
axes[1].text(0, -1.3, 'Natal', ha='center', fontsize=14)

# Título geral
plt.suptitle('Proporção de registros com 0 casos de Dengue', fontsize=16, y=1.05)

plt.tight_layout()
plt.savefig('percentual_zeros.png')
