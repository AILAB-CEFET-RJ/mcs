# Create map of cases and weather stations

import pandas as pd
import folium
from folium.plugins import HeatMap

df_eac = pd.read_csv('cea.csv')
df = pd.read_parquet('data/processed/sinan/sinan.parquet')

agg_df = df.groupby('ID_UNIDADE').agg({
    'count': 'sum',  # Sum the case counts
    'lat': 'first',  # Assuming lat/lon are consistent, otherwise consider 'mean'
    'lon': 'first'   # Assuming lat/lon are consistent, otherwise consider 'mean'
}).reset_index()

lat_center = -22.3534
lon_center = -42.7076

mapa = folium.Map(location=[lat_center, lon_center], zoom_start=10)

for _, row in agg_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"CNES: {row['ID_UNIDADE']}<br>Casos: {row['count']}",
        icon=folium.Icon(color="red")
    ).add_to(mapa)

for _, row in df_eac.iterrows():
    folium.Marker(
        location=[row['VL_LATITUDE'], row['VL_LONGITUDE']],
        popup=f"Estação: {row['DC_NOME']}<br>Situação: {row['CD_SITUACAO']}",
        icon=folium.Icon(color="blue")
    ).add_to(mapa)    

mapa.save('mapa_casos.html')