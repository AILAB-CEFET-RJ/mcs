# Create INMET Dataframe
import pandas as pd

# Rio de Janeiro extent
loni = -44.98
lonf = -23.79
lati = -40.92
latf = -20.27

df = pd.read_csv('/run/media/metatron/Yggdrasil/python/arboseer/stations.csv', delimiter=";", decimal=",")
df = df[(df['VL_LONGITUDE'] >= loni) & (df['VL_LONGITUDE'] <= lonf) &
                 (df['VL_LATITUDE'] >= lati) & (df['VL_LATITUDE'] <= latf)]
df.to_csv('/run/media/metatron/Yggdrasil/python/arboseer/stations.csv')
df = pd.read_csv('/run/media/metatron/Yggdrasil/python/arboseer/stations.csv')

colunas_a_manter = ['DC_NOME', 'SG_ESTADO', 'VL_LATITUDE', 'VL_LONGITUDE', 'DT_INICIO_OPERACAO', 'CD_ESTACAO']
df = df[colunas_a_manter]

df.to_csv('/run/media/metatron/Yggdrasil/python/arboseer/stations.csv')