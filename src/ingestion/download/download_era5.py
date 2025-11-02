#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baixa ERA5-Land (horário) mês a mês no formato NetCDF.

Uso:
  python era5_land_downloader.py \
    --start-year 2016 --end-year 2019 \
    --area "-5.45,-35.6,-6.15,-34.85" \
    --file-prefix NATAL \
    --out-dir ./ERA5

Observações:
- 'area' deve estar no formato: norte, oeste, sul, leste (N, W, S, E).
- Pula automaticamente arquivos que já existem (use --overwrite para substituir).
"""

import argparse
import os
from pathlib import Path
import time
import calendar
import cdsapi


DEFAULT_VARIABLES = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "total_precipitation",
]

ALL_HOURS = [f"{h:02d}:00" for h in range(24)]

DATASET = "reanalysis-era5-land"  # mantém o dataset do seu exemplo


def parse_area(area_str):
    """
    Converte string "N,W,S,E" (com vírgula ou espaço) para lista [N, W, S, E] de floats.
    """
    # Permite vírgula OU espaço como separador
    raw = area_str.replace(",", " ").split()
    if len(raw) != 4:
        raise ValueError("Área deve ter exatamente 4 valores: N, W, S, E")
    try:
        north, west, south, east = map(float, raw)
    except Exception as exc:
        raise ValueError(f"Não foi possível converter 'area' para floats: {exc}")
    return [north, west, south, east]


def days_in_month(year, month):
    """
    Retorna lista de strings 'DD' com os dias válidos do mês/ano.
    (Respeita anos bissextos, evita pedir dia 31 em meses com 30 etc.)
    """
    _, num_days = calendar.monthrange(year, month)
    return [f"{d:02d}" for d in range(1, num_days + 1)]


def month_str(month_int):
    return f"{month_int:02d}"


def safe_retrieve(client, dataset, request, target_path, max_retries=5, base_sleep=5):
    """
    Faz retrieve com tentativas e recuo exponencial simples.
    Salva direto em target_path (sem .download() separado).
    """
    attempt = 0
    while True:
        try:
            client.retrieve(dataset, request, target=str(target_path))
            return
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"[WARN] Falha ao baixar ({e}). Tentando novamente em {sleep_s}s "
                  f"({attempt}/{max_retries})...")
            time.sleep(sleep_s)


def build_request(year, month, area, variables):
    """
    Monta o dicionário de request para o CDS API.
    """
    req = {
        "variable": variables,
        "year": str(year),
        "month": month_str(month),
        "day": days_in_month(year, month),
        "time": ALL_HOURS,
        # A API do CDS normalmente usa 'format' para NetCDF
        "format": "netcdf",
        # Mantém download sem arquivar (útil quando disponível)
        "download_format": "unarchived",
        "area": area,  # [N, W, S, E]
    }
    return req


def main():
    parser = argparse.ArgumentParser(
        description="Baixa ERA5-Land horário por mês, salvando como PREFIXO_ANO_MES.nc"
    )
    parser.add_argument("--start-year", type=int, required=True, help="Ano inicial (ex: 2016)")
    parser.add_argument("--end-year", type=int, required=True, help="Ano final (ex: 2019, inclusive)")
    parser.add_argument("--area", type=str, required=True,
                        help="Área no formato 'N,W,S,E' (pode usar espaços em vez de vírgulas)")
    parser.add_argument("--file-prefix", type=str, required=True, help="Prefixo do arquivo (ex: NATAL)")
    parser.add_argument("--out-dir", type=str, required=True, help="Diretório de saída")
    parser.add_argument("--overwrite", action="store_true", help="Rebaixar e sobrescrever arquivos existentes")
    # Opcional: permitir trocar as variáveis mantendo padrão do seu script original
    parser.add_argument("--variables", type=str, default=",".join(DEFAULT_VARIABLES),
                        help=f"Lista de variáveis separadas por vírgula. Padrão: {', '.join(DEFAULT_VARIABLES)}")

    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("start-year deve ser menor ou igual a end-year.")

    area = parse_area(args.area)
    variables = [v.strip() for v in args.variables.replace(";", ",").split(",") if v.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    total = (args.end_year - args.start_year + 1) * 12
    count = 0

    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            count += 1
            filename = f"{args.file_prefix}_{year}_{month:02d}.nc"
            target_path = out_dir / filename

            if target_path.exists() and not args.overwrite:
                print(f"[SKIP] Já existe: {target_path}")
                continue

            request = build_request(year, month, area, variables)

            print(f"[{count}/{total}] Baixando {filename} ...")
            try:
                safe_retrieve(client, DATASET, request, target_path)
                # Verificação simples pós-download
                if target_path.exists() and target_path.stat().st_size > 0:
                    print(f"[OK] Salvo em: {target_path}")
                else:
                    raise RuntimeError("Download concluído mas arquivo parece vazio.")
            except Exception as e:
                print(f"[ERRO] Falha ao baixar {filename}: {e}")


if __name__ == "__main__":
    main()
