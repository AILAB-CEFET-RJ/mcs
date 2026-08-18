# Trainer STConv para comparação tabular × espaço-temporal

O trainer usa exatamente os cortes já materializados no diretório tensorial.
Ele não refaz nem mistura treino, validação e teste.

## Contrato do dataset

Cada diretório (por exemplo, `RJ_DAILY_FULL_STCONV`) contém:

- `train_features.npy` ou `train_features_by_year/train_features_*.npy`;
- `val_features.npy` e `test_features.npy`;
- `{split}_targets.npy`;
- `{split}_dates.npy`;
- `channels.json`.

As features têm forma `(tempo, canais, altura, largura)` e os alvos
`(tempo, altura, largura)`.

## Treino

Exemplo diário com 28 passos de entrada e um passo de saída:

```powershell
python arboseer/src/train_stconv.py `
  --dataset-dir arboseer/data/datasets/RJ_DAILY_FULL_STCONV `
  --input-steps 28 `
  --output-steps 1 `
  --model official-r `
  --loss poisson_nll_log `
  --epochs 50 `
  --batch-size 4
```

Para o semanal, `input-steps` e `output-steps` são expressos em semanas. Para
comparação com o tabular de previsão do próximo período, use `output-steps 1`.

## Fusão meteorológica

O ERA5 é obrigatório e constitui a cobertura completa. AlertaRio,
WebSirenes e aeroportos são opcionais. A ordem dos argumentos define a
prioridade:

```powershell
python arboseer/src/data_handling/stations/fuse_weather_sources.py `
  --era5 arboseer/data/processed/weather/RJ_era5_daily.npz `
  --out arboseer/data/processed/weather/RJ_fused_daily.npz `
  --source ALERTARIO=arboseer/data/processed/weather/alertario_daily.csv `
  --source WEBSIRENES=arboseer/data/processed/weather/websirenes_daily.csv `
  --source AIRPORTS=arboseer/data/processed/weather/airports_daily.csv
```

Se uma fonte ou uma observação não existir, o valor da fonte seguinte é
tentado; ao final, permanece o ERA5. O NPZ inclui `source_code`, e o arquivo
`.provenance.json` registra cobertura e prioridade, permitindo relatar a
composição efetiva dos dados.

Os CSVs opcionais devem estar agregados na mesma resolução do ERA5 e conter
`DATE, STATION, LAT, LNG`, além dos canais disponíveis entre
`TEM_AVG, TEM_MIN, TEM_MAX, DEW_AVG, RH_AVG/HUM_AVG, RAIN/PRECIP`.

