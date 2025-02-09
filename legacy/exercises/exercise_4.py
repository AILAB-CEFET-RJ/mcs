# Plot cases x temperature by year

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Configuração inicial
months_ordered = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
years = ['2019','2020','2021','2022','2023']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Função para processar datasets de dengue
def process_dengue_dataset(parquet_path, yy):
    ds = pd.read_parquet(parquet_path)
    #ds = ds[ds["SG_UF"] == '33']
    ds = ds[ds["ID_UNIDADE"].isin(['7998678', '2273411', '0697524', '3416356', '2298120'])]
    ds = ds.groupby(["ID_AGRAVO", "ID_UNIDADE", "DT_NOTIFIC"]).size().reset_index(name="Cases")
    ds['DT_NOTIFIC'] = pd.to_datetime(ds['DT_NOTIFIC'], format='%Y%m%d')
    ds['Year'] = ds['DT_NOTIFIC'].dt.year
    ds['Month'] = ds['DT_NOTIFIC'].dt.month
    ds = ds[ds["Year"] == 2000+yy]
    ds = ds.groupby(['Year', 'Month']).agg(Cases=('Cases', 'sum')).reset_index()      

    return ds

def process_temperature_dataset(parquet_path):
    ds = pd.read_parquet(parquet_path)
    ds['DT_MEDICAO'] = pd.to_datetime(ds['DT_MEDICAO'])
    ds['Year'] = ds['DT_MEDICAO'].dt.year
    ds['Month'] = ds['DT_MEDICAO'].dt.month
    ds = ds.groupby(['Year', 'Month']).agg(
        TEM_MIN=('TEM_MIN', 'min'),
        TEM_MAX=('TEM_MAX', 'max'),
        TEM_AVG=('TEM_INS', 'mean')
    ).reset_index()
    return ds

dengue_data = pd.concat([process_dengue_dataset(f'data/raw/sinan/DENGBR{year}.parquet', year) for year in [19, 20, 21, 22, 23]])
ws_ds = process_temperature_dataset("data/processed/inmet/A621.parquet")    

temp_avg_annual = ws_ds.pivot(index='Month', columns='Year', values='TEM_AVG')
dengue_cases_annual = dengue_data.pivot(index='Month', columns='Year', values='Cases')

width = 0.1
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plotar as temperaturas médias novamente
for (year, color) in zip(temp_avg_annual.columns, colors):
    ax1.plot(temp_avg_annual.index, temp_avg_annual[year], label=f'Temperatura {year}', marker='o', color=color)
ax1.set_xlabel('Mês')
ax1.set_ylabel('Temperatura Média (°C)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_title('Temperatura Média e Casos de Dengue por Mês')

# Criar eixo y secundário para os casos de dengue novamente
ax2 = ax1.twinx()
for (idx, year) in enumerate(dengue_cases_annual.columns):
    ax2.bar(dengue_cases_annual.index + width*idx - width*(len(dengue_cases_annual.columns)-1)/2, dengue_cases_annual[year], width=width, label=f'Casos Dengue {year}', alpha=0.7)

ax2.set_ylabel('Casos de Dengue', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

lines, labels = ax1.get_legend_handles_labels()
bars, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + bars, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.xticks(np.arange(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.tight_layout()
plt.savefig(f'data/processed/temp_cases_5years.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()