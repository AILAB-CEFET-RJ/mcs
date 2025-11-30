import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# ---------------------------------------------
# 1. Carregar shapefile do polígono do estado
# ---------------------------------------------
state_shape = gpd.read_file('data/shapes/RJ_Municipios_2024/RJ_Municipios_2024.shp')
#state_shape = gpd.read_file('data/shapes/Natal/Limite_Bairros.shp')
state_shape = state_shape.to_crs(epsg=3857)

# ---------------------------------------------
# 2. Carregar o Parquet com todas as unidades
# ---------------------------------------------
df_cnes = pd.read_parquet('data/processed/cnes/STRJ2401.parquet')
#df_cnes = pd.read_parquet('data/processed/cnes/SRN2401.parquet')
print("CNES", df_cnes.columns)

# ---------------------------------------------
# 3. Carregar o Parquet com os casos de Dengue
# ---------------------------------------------
df_dengue = pd.read_parquet('data/processed/sinan/DENG33.parquet')
#df_dengue = pd.read_parquet('data/processed/sinan/DENG240810.parquet')
print("DENGUE", df_dengue.columns)

# ---------------------------------------------
# 4. Filtrar apenas unidades presentes no SINAN
# ---------------------------------------------
# Checar as colunas corretas: normalmente ID_UNIDADE no dengue e CNES no cnes
id_unidades_dengue = df_dengue['ID_UNIDADE'].dropna().unique()

# Filtrar o DF CNES para manter só as unidades presentes nos dados de casos
df_cnes_filtrado = df_cnes[df_cnes['CNES'].isin(id_unidades_dengue)].copy()

# ---------------------------------------------
# 5. Remover nulos em LAT/LNG
# ---------------------------------------------
df_cnes_filtrado = df_cnes_filtrado.dropna(subset=['LAT', 'LNG']).drop_duplicates()

# ---------------------------------------------
# 6. Criar GeoDataFrame das Unidades
# ---------------------------------------------
geometry_units = [Point(xy) for xy in zip(df_cnes_filtrado['LNG'], df_cnes_filtrado['LAT'])]
gdf_units = gpd.GeoDataFrame(df_cnes_filtrado, geometry=geometry_units, crs='EPSG:4326')
gdf_units = gdf_units.to_crs(epsg=3857)

# ---------------------------------------------
# 7. Contar número de unidades válidas
# ---------------------------------------------
total_unidades = len(gdf_units)

# ---------------------------------------------
# 8. Plotar o Mapa
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_axis_off()

# Estado
state_shape.plot(ax=ax, edgecolor='black', facecolor='none')

# Unidades filtradas
gdf_units.plot(ax=ax, color='blue', marker='o', markersize=30, label='Unidades de Saúde')

# Ajustar limites com folga
ax.set_xlim(state_shape.total_bounds[0] - 5000, state_shape.total_bounds[2] + 5000)
ax.set_ylim(state_shape.total_bounds[1] - 5000, state_shape.total_bounds[3] + 5000)

# Título com contagem
plt.title(f"Unidades de Saúde no Estado do RJ (Usadas no Dataset de Dengue, Total: {total_unidades})", fontsize=14)
#plt.title(f"Unidades de Saúde da Cidade de Natal (Usadas no Dataset de Dengue, Total: {total_unidades})", fontsize=14)

plt.legend()
plt.tight_layout()
plt.savefig('mapa_unidades_saude_rj.png')
#plt.savefig('mapa_unidades_saude_rn.png')