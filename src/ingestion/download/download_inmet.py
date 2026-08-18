#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download de dados horários automáticos do INMET via API token.

Uso:
  python src/ingestion/download/download_inmet.py \
    --token "TOKEN" --city RJ --start 2014 --end 2023

Saída: data/raw/inmet/{STATION_ID}.csv  (um arquivo por estação)
       Pula arquivos que já existem.
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

INMET_API_BASE_URL = "https://apitempo.inmet.gov.br"

STATIONS_BY_CITY = {
    "RJ": [
        {"id": "A601", "name": "Seropédica"},
        {"id": "A602", "name": "Maramabaias"},
        {"id": "A603", "name": "Duque de Caxias - Xerém"},
        {"id": "A604", "name": "Cambuci"},
        {"id": "A606", "name": "Arraial do Cabo"},
        {"id": "A607", "name": "Campos dos Goytacazes"},
        {"id": "A608", "name": "Macaé"},
        {"id": "A609", "name": "Resende"},
        {"id": "A610", "name": "Pico do Couto"},
        {"id": "A611", "name": "Valença"},
        {"id": "A618", "name": "Teresópolis - Parque Nacional"},
        {"id": "A619", "name": "Paraty"},
        {"id": "A620", "name": "Campos dos Goytacazes - São Tomé"},
        {"id": "A621", "name": "Vila Militar"},
        {"id": "A624", "name": "Nova Friburgo - Salinas"},
        {"id": "A625", "name": "Três Rios"},
        {"id": "A626", "name": "Rio Claro"},
        {"id": "A627", "name": "Niterói"},
        {"id": "A628", "name": "Angra dos Reis"},
        {"id": "A629", "name": "Carmo"},
        {"id": "A630", "name": "Santa Maria Madalena"},
        {"id": "A636", "name": "Jacarepaguá"},
        {"id": "A637", "name": "Paty do Alferes - Avelar"},
        {"id": "A652", "name": "Forte de Copacabana"},
        {"id": "A659", "name": "Silva Jardim"},
        {"id": "A667", "name": "Saquarema - Sampaio Correia"},
        # Estações de MG explicitamente incluídas por decisão metodológica.
        {"id": "A518", "name": "Juiz de Fora"},
        {"id": "A557", "name": "Coronel Pacheco"},
    ]
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_json(session: requests.Session, url: str, timeout=(10, 120)):
    try:
        resp = session.get(url, timeout=timeout)
    except requests.RequestException as e:
        logging.warning("Erro de rede: %s", e)
        return None

    if resp.status_code != 200:
        logging.warning("HTTP %s para %s  Body(<=300c): %.300s", resp.status_code, url, resp.text)
        return None

    ct = (resp.headers.get("Content-Type", "") or "").lower()
    if "json" not in ct:
        logging.warning(
            "Resposta não-JSON (Content-Type=%s) para %s  Body(<=300c): %.300s",
            resp.headers.get("Content-Type"),
            url,
            resp.text,
        )
        return None

    try:
        return resp.json()
    except ValueError:
        logging.warning("JSON inválido de %s  Body(<=300c): %.300s", url, resp.text)
        return None


def download_station(
    session: requests.Session,
    station_id: str,
    start: int,
    end: int,
    token: str,
    out_dir: Path,
):
    out_file = out_dir / f"{station_id}.csv"
    if out_file.exists():
        logging.info("[SKIP] %s já existe: %s", station_id, out_file)
        return

    station_safe = quote(station_id, safe="")
    token_safe = quote(token, safe="")
    now = datetime.now()
    end = min(end, now.year)

    dfs = []
    for year in range(start, end + 1):
        date_ini = f"{year}-01-01"
        date_fin = f"{year}-12-31" if year < now.year else now.strftime("%Y-%m-%d")
        url = (
            f"{INMET_API_BASE_URL}/token/estacao"
            f"/{date_ini}/{date_fin}/{station_safe}/{token_safe}"
        )
        logging.info("[%s] %d  GET %s", station_id, year, url)

        data = get_json(session, url)
        if not data:
            logging.warning("[%s] Sem dados para %d", station_id, year)
            time.sleep(1)
            continue

        df = pd.DataFrame(data)
        dfs.append(df)
        logging.info("[%s] %d → %d registros", station_id, year, len(df))
        time.sleep(0.5)

    if not dfs:
        logging.warning("[%s] Nenhum dado baixado — arquivo não criado.", station_id)
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(out_file, index=False)
    logging.info("[%s] Salvo: %s (%d linhas)", station_id, out_file, len(df_all))


def build_malha1(out_dir: Path) -> None:
    """Aggregate all A6*.csv hourly files in out_dir into malha1_inmet_diario.csv."""
    import glob as _glob
    import numpy as np

    station_files = sorted(_glob.glob(str(out_dir / "A6*.csv")))
    if not station_files:
        raise FileNotFoundError(f"No A6*.csv files found in {out_dir}")

    daily_frames = []
    for fpath in station_files:
        logging.info("[malha1] Processing %s", fpath)
        df = pd.read_csv(fpath, encoding="latin1")

        # Coordinates and station metadata (constant per file)
        lat = pd.to_numeric(df["VL_LATITUDE"], errors="coerce").iloc[0]
        lng = pd.to_numeric(df["VL_LONGITUDE"], errors="coerce").iloc[0]
        station_id = df["CD_ESTACAO"].iloc[0]
        name = df["DC_NOME"].iloc[0] if "DC_NOME" in df.columns else station_id

        df["DT_MEDICAO"] = pd.to_datetime(df["DT_MEDICAO"], errors="coerce")
        for col in ["TEM_INS", "TEM_MIN", "TEM_MAX", "CHUVA", "UMD_INS", "PTO_INS", "PRE_INS"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        grp = df.groupby("DT_MEDICAO")
        daily = pd.DataFrame({
            "DT_MEDICAO": grp["DT_MEDICAO"].first().index,
            "TEM_AVG":  grp["TEM_INS"].mean().values,
            "TEM_MIN":  grp["TEM_INS"].min().values,
            "TEM_MAX":  grp["TEM_INS"].max().values,
            "HUM_AVG":  grp["UMD_INS"].mean().values,
            "DEW_AVG":  grp["PTO_INS"].mean().values,
            "RAIN":     grp["CHUVA"].sum().values,
            "PRES_AVG": grp["PRE_INS"].mean().values if "PRE_INS" in df.columns else np.nan,
            "N_OBS":    grp["TEM_INS"].count().values,
        })
        daily["STATION"] = station_id
        daily["NAME"] = name
        daily["CITY"] = ""
        daily["LAT"] = lat
        daily["LNG"] = lng
        daily["VV_AVG"] = np.nan
        daily["VD_AVG"] = np.nan
        daily_frames.append(daily)

    result = pd.concat(daily_frames, ignore_index=True)
    result["DT_MEDICAO"] = result["DT_MEDICAO"].dt.strftime("%Y-%m-%d")
    col_order = [
        "DT_MEDICAO", "STATION", "NAME", "CITY", "LAT", "LNG",
        "TEM_MIN", "TEM_MAX", "TEM_AVG", "HUM_AVG", "DEW_AVG",
        "RAIN", "VV_AVG", "VD_AVG", "PRES_AVG", "N_OBS",
    ]
    result = result[col_order].sort_values(["STATION", "DT_MEDICAO"]).reset_index(drop=True)
    out_path = out_dir / "malha1_inmet_diario.csv"
    result.to_csv(out_path, index=False)
    logging.info("[malha1] Saved %s (%d rows, %d stations)", out_path, len(result), result["STATION"].nunique())


def main():
    parser = argparse.ArgumentParser(
        description="Download de dados horários automáticos do INMET"
    )
    parser.add_argument("--malha1-only", action="store_true",
                        help="Apenas agregar CSVs existentes em malha1_inmet_diario.csv")
    parser.add_argument("--token", default=None, help="Token da API INMET")
    parser.add_argument("--city", default=None, help="Código da cidade (ex: RJ)")
    parser.add_argument("--start", type=int, default=None, help="Ano inicial")
    parser.add_argument("--end", type=int, default=None, help="Ano final (inclusive)")
    parser.add_argument("--out-dir", default="data/raw/inmet", help="Diretório de saída")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.malha1_only:
        build_malha1(out_dir)
        return

    if not args.token or not args.city or args.start is None or args.end is None:
        parser.error("--token, --city, --start, --end são obrigatórios para download")

    city = args.city.upper()
    if city not in STATIONS_BY_CITY:
        raise ValueError(
            f"Cidade '{city}' não configurada. Disponíveis: {list(STATIONS_BY_CITY)}"
        )

    session = build_session()
    stations = STATIONS_BY_CITY[city]
    logging.info("Iniciando download INMET — %d estações para %s (%d–%d)", len(stations), city, args.start, args.end)

    for st in stations:
        download_station(session, st["id"], args.start, args.end, args.token, out_dir)

    logging.info("Download INMET concluído.")


if __name__ == "__main__":
    main()
