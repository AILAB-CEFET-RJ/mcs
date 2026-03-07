#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from datetime import datetime
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

INMET_API_BASE_URL = "https://apitempo.inmet.gov.br"

# Se quiser manter sua lista, pode deixar aqui.
INMET_WEATHER_STATION_IDS = (
    "A304",
    # "A621",
)

DEFAULT_HEADERS = {
    # Ajuda a evitar 403 (WAF/CDN) quando a API bloqueia clients “genéricos”
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


def get_json_df(session: requests.Session, url: str, timeout=(10, 60)) -> pd.DataFrame:
    """
    Faz GET com headers + retry + timeout e devolve DataFrame.
    Em qualquer erro (rede, 403, não-JSON, JSON inválido), devolve DataFrame vazio.
    """
    try:
        resp = session.get(url, timeout=timeout)
    except requests.RequestException as e:
        logging.exception("Erro de rede ao chamar INMET: %s", e)
        return pd.DataFrame()

    if resp.status_code != 200:
        ct = resp.headers.get("Content-Type", "")
        snippet = (resp.text or "")[:300].replace("\n", " ")
        logging.warning(
            "INMET respondeu %s (Content-Type=%s) para %s. Body(<=300c): %s",
            resp.status_code,
            ct,
            url,
            snippet,
        )
        return pd.DataFrame()

    # Protege contra casos em que volta HTML mesmo com 200
    ct_lower = (resp.headers.get("Content-Type", "") or "").lower()
    if "json" not in ct_lower:
        snippet = (resp.text or "")[:300].replace("\n", " ")
        logging.warning(
            "Resposta 200 mas não parece JSON (Content-Type=%s) para %s. Body(<=300c): %s",
            resp.headers.get("Content-Type", ""),
            url,
            snippet,
        )
        return pd.DataFrame()

    try:
        data = resp.json()
    except ValueError:
        snippet = (resp.text or "")[:300].replace("\n", " ")
        logging.warning("Falha ao decodificar JSON para %s. Body(<=300c): %s", url, snippet)
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def retrieve_from_station(
    session: requests.Session,
    station_id: str,
    beginning_year: int,
    end_year: int,
    api_token: str,
    output_path: str,
):
    os.makedirs(output_path, exist_ok=True)

    now = datetime.now()
    current_year = now.year
    end_year = min(end_year, current_year)

    # >>> IMPORTANTE: não usar pd.read_json(URL) porque pode dar 403 (como você viu)
    df_inmet_stations = get_json_df(session, f"{INMET_API_BASE_URL}/estacoes/T")
    if df_inmet_stations.empty:
        raise RuntimeError("Falhou ao baixar lista de estações (/estacoes/T). Veja os logs (403 etc.).")

    station_row = df_inmet_stations[df_inmet_stations["CD_ESTACAO"] == station_id]
    if station_row.empty:
        raise ValueError(f"Estação '{station_id}' não encontrada em /estacoes/T")

    logging.info("Downloading observations from weather station %s...", station_id)

    # Escapa token/id para não quebrar URL caso tenha caractere especial
    station_safe = quote(station_id, safe="")
    token_safe = quote(api_token, safe="")

    dfs = []
    for year in range(beginning_year, end_year + 1):
        start = f"{year}-01-01"
        if year == end_year:
            end = min(datetime.strptime(f"{year}-12-31", "%Y-%m-%d"), now).strftime("%Y-%m-%d")
        else:
            end = f"{year}-12-31"

        url = f"{INMET_API_BASE_URL}/token/estacao/{start}/{end}/{station_safe}/{token_safe}"
        logging.info("GET %s", url)

        df_year = get_json_df(session, url)
        if df_year.empty:
            logging.info("Sem dados (ou falha) para %s em %s..%s", station_id, start, end)

        dfs.append(df_year)

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    filename = os.path.join(output_path, f"{station_row['CD_ESTACAO'].iloc[0]}.parquet")
    logging.info("Saving to '%s' (linhas=%s)", filename, len(df_all))
    df_all.to_parquet(filename, index=False)


def retrieve_data(station_id: str, initial_year: int, final_year: int, api_token: str, output_path: str):
    session = build_session()

    if station_id == "all":
        df_inmet_stations = get_json_df(session, f"{INMET_API_BASE_URL}/estacoes/T")
        if df_inmet_stations.empty:
            raise RuntimeError("Falhou ao baixar lista de estações (/estacoes/T). Veja os logs.")
        station_rows = df_inmet_stations[df_inmet_stations["CD_ESTACAO"].isin(INMET_WEATHER_STATION_IDS)]

        for sid in station_rows["CD_ESTACAO"].tolist():
            retrieve_from_station(session, sid, initial_year, final_year, api_token, output_path)
    else:
        retrieve_from_station(session, station_id, initial_year, final_year, api_token, output_path)


def main(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        usage="{0} -s <ws_id> -b <begin_year> -e <end_year> -t <api_token> -o <output_path>".format(argv[0]),
        description="Download de observações da API do INMET e salva em parquet.",
    )
    parser.add_argument("-t", "--api_token", required=True, help="INMET API token", metavar="")
    parser.add_argument("-s", "--ws_id", required=True, help="Weather station ID (ex: A621 ou all)", metavar="")
    parser.add_argument("-b", "--begin_year", type=int, required=True, help="Start year", metavar="")
    parser.add_argument("-e", "--end_year", type=int, required=True, help="End year", metavar="")
    parser.add_argument("-o", "--output_path", required=True, help="Output path", metavar="")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parser.parse_args(argv[1:])

    api_token = args.api_token
    station_id = args.ws_id
    start_year = args.begin_year
    end_year = args.end_year
    output_path = args.output_path

    if not api_token:
        parser.error("api_token vazio")
    if not station_id:
        parser.error("ws_id vazio")
    if start_year > end_year:
        parser.error("begin_year deve ser <= end_year")

    retrieve_data(station_id, start_year, end_year, api_token, output_path)


if __name__ == "__main__":
    main(sys.argv)