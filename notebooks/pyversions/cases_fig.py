import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1. Carregar dados
# ------------------------------
df = pd.read_parquet('data/processed/sinan/DENG33.parquet')
#df = pd.read_parquet('data/processed/sinan/DENG240810.parquet')

# ------------------------------
# 2. Converter data
# ------------------------------
df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'])

# ------------------------------
# 3. Agregar casos por dia
# ------------------------------
df_agg = df.groupby('DT_NOTIFIC')['CASES'].sum().reset_index()

# ------------------------------
# 4. Calcular total de casos
# ------------------------------
total_casos = df_agg['CASES'].sum()

# ------------------------------
# 5. Plotar gráfico
# ------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df_agg['DT_NOTIFIC'], df_agg['CASES'], color='blue', lw=1.5)
plt.title(f'Casos Diários de Dengue - Cidade de Natal (Total: {total_casos:,} casos)')
plt.xlabel('Data da Notificação')
plt.ylabel('Número de Casos')
plt.grid(True, linestyle='--', alpha=0.5)

# Adicionar anotação discreta
plt.text(0.99, 0.01, f'Total: {total_casos:,} casos',
         transform=plt.gca().transAxes,
         fontsize=10, color='gray',
         ha='right', va='bottom')

plt.tight_layout()
plt.savefig('casos_rj.png')
#plt.savefig('casos_natal.png')
#plt.show()
