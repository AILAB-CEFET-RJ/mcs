import pandas as pd

df = pd.read_parquet("data/processed/inmet/aggregated.parquet")
df.to_csv('teste.csv')