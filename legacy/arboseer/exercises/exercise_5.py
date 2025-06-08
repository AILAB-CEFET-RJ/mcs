import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Carregar o shapefile do estado do Rio de Janeiro
state_shape = gpd.read_file('data/shapes/RJ_Municipios_2022/RJ_Municipios_2022.shp')

# Carregar o arquivo Parquet com as unidades
df = pd.read_parquet('data/processed/sinan/sinan.parquet')

# Remover duplicações nas colunas de LAT e LNG
df_unique_units = df[['LAT', 'LNG']].drop_duplicates()

# Criar uma coluna de geometria para as unidades (LAT, LNG)
geometry_units = [Point(xy) for xy in zip(df_unique_units['LNG'], df_unique_units['LAT'])]

# Criar um GeoDataFrame usando as latitudes e longitudes das unidades
gdf_units = gpd.GeoDataFrame(df_unique_units, geometry=geometry_units)

# Remover duplicações nas colunas closest_LAT_INMET e closest_LNG_INMET
df_unique_inmet = df[['closest_LAT_INMET', 'closest_LNG_INMET']].drop_duplicates()

# Criar uma coluna de geometria para as estações INMET (closest_LAT_INMET, closest_LNG_INMET)
geometry_inmet = [Point(xy) for xy in zip(df_unique_inmet['closest_LNG_INMET'], df_unique_inmet['closest_LAT_INMET'])]

# Criar um GeoDataFrame usando as latitudes e longitudes das estações INMET
gdf_inmet = gpd.GeoDataFrame(df_unique_inmet, geometry=geometry_inmet)

# Definir o sistema de referência espacial (SIRGAS 2000, EPSG:4326) para ambos os GeoDataFrames
gdf_units.set_crs(epsg=4326, inplace=True)
gdf_inmet.set_crs(epsg=4326, inplace=True)

# Converter o shapefile e os GeoDataFrames para a projeção métrica (Mercator)
state_shape = state_shape.to_crs(epsg=3857)
gdf_units = gdf_units.to_crs(epsg=3857)
gdf_inmet = gdf_inmet.to_crs(epsg=3857)

# Criar o mapa
fig, ax = plt.subplots(figsize=(10, 10))

# Remover os eixos com as coordenadas
ax.set_axis_off()

# Plotar o shapefile do estado do Rio de Janeiro
state_shape.plot(ax=ax, edgecolor='black', facecolor='none')

# Plotar os pontos das unidades em azul
gdf_units.plot(ax=ax, color='blue', marker='o', markersize=50, label='Unidades')

# Plotar os pontos mais próximos das estações INMET em vermelho
gdf_inmet.plot(ax=ax, color='red', marker='x', markersize=50, label='Estações INMET')

# Ajustar os limites do mapa com base nos limites do shapefile
ax.set_xlim(state_shape.total_bounds[0] - 5000, state_shape.total_bounds[2] + 5000)
ax.set_ylim(state_shape.total_bounds[1] - 5000, state_shape.total_bounds[3] + 5000)

# Ajustar o layout para ficar mais compacto
plt.tight_layout()

# Adicionar título e legenda
plt.legend()
plt.title("Unidades e Estações INMET no Estado do Rio de Janeiro")
plt.show()
