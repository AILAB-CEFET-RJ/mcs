# Plot SAT x WS Avg, Max, Min

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
from src.utils.reproject_lats_lons import reproject_lats_lons
from src.utils.find_closest_lat_lon import find_closest_lat_lon
import matplotlib.dates as mdates

lst_reference_file = "data/raw/lst/20230116/OR_ABI-L2-LSTF-M6_G16_s20230152100206_e20230152109514_c20230152111229.nc"
lons, lats = reproject_lats_lons(lst_reference_file)

lat = -22.940000
lon = -43.402778

x, y = find_closest_lat_lon(lat, lon, lats, lons)
dir_path = 'data/processed/lst'

def process_temperature_dataset(parquet_path):
    ds = pd.read_parquet(parquet_path)
    ds['DT_MEDICAO'] = pd.to_datetime(ds['DT_MEDICAO'])
    ds['Year'] = ds['DT_MEDICAO'].dt.year
    ds['Month'] = ds['DT_MEDICAO'].dt.month
    ds = ds.groupby(['Year', 'Month']).agg(
        min=('TEM_MIN', 'min'),
        max=('TEM_MAX', 'max'),
        avg=('TEM_INS', 'mean')
    ).reset_index()

    ds = ds[ds['Year'] == 2023]
    return ds

ws_df = process_temperature_dataset("data/processed/inmet/A636.parquet")   

sat_df = pd.DataFrame(columns=['Year', 'Month', 'Avg', 'Max', 'Min'])

# Listar todos os arquivos .npz no diretório
arquivos = [f for f in os.listdir(dir_path) if f.endswith('.npz') and len(f) == 10]
arquivos.sort()

for arquivo in arquivos:
    year, month = arquivo[:4], arquivo[4:6]
    
    if month[0] == '0':
        month = month[1]
    
    data = np.load(os.path.join(dir_path, arquivo))
    
    avg_val = data['avg'][x, y]
    max_val = data['max'][x, y]
    min_val = data['min'][x, y]
    
    new_row = {'Year': year, 'Month': month, 'Avg': avg_val, 'Max': max_val, 'Min': min_val}
    sat_df = pd.concat([sat_df, pd.DataFrame([new_row])], ignore_index=True)

#sat_ds.to_csv('sat.csv', index=False)
#ws_ds.to_csv('ws.csv', index=False)

sat_df['Date'] = pd.to_datetime(sat_df[['Year', 'Month']].assign(DAY=1))
ws_df['Date'] = pd.to_datetime(ws_df[['Year', 'Month']].assign(DAY=1))

plt.figure(figsize=(14, 7))

# Plot SAT Avg, Max, Min
plt.plot(sat_df['Date'], sat_df['Avg'], label='Satellite Avg', marker='o', linestyle='-', color='blue')
plt.plot(sat_df['Date'], sat_df['Max'], label='Satellite Max', marker='^', linestyle='--', color='lightblue')
plt.plot(sat_df['Date'], sat_df['Min'], label='Satellite Min', marker='v', linestyle='--', color='darkblue')

# Plot WS Avg, Max, Min
plt.plot(ws_df['Date'], ws_df['avg'], label='Weather Station Avg', marker='x', linestyle='-', color='red')
plt.plot(ws_df['Date'], ws_df['max'], label='Weather Station Max', marker='^', linestyle='--', color='pink')
plt.plot(ws_df['Date'], ws_df['min'], label='Weather Station Min', marker='v', linestyle='--', color='darkred')

plt.title('Satellite vs Weather Station')
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))

plt.gcf().autofmt_xdate()
plt.savefig(f'data/processed/ws_vs_sat_A636.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()



