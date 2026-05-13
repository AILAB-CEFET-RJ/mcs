#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download de dados METAR das estações aeroportuárias via API REDEMET.

Uso:
  python src/ingestion/download/download_redemet.py \
    --city RJ --start 2014-01-01 --end 2023-12-31

Saída: data/raw/redemet/{ICAO}.csv  (um arquivo por estação)
       Pula arquivos que já existem.
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REDEMET_API_BASE_URL = "https://api-redemet.decea.mil.br"
REDEMET_API_KEY = "aOrV6N5DtQen69uEDlYwbUTo3CTTCcAZ50r7WEW4"
PAGE_SIZE = 100

STATIONS_BY_CITY = {
    "RJ": [
        {"icao": "SBGL", "name": "Galeão"},
        {"icao": "SBRJ", "name": "Santos Dumont"},
        {"icao": "SBSC", "name": "Santa Cruz"},
        {"icao": "SBAF", "name": "Campo dos Afonsos"},
    ]
}


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_metar_page(
    session: requests.Session,
    icao: str,
    date_ini: str,
    date_fim: str,
    page: int = 1,
) -> dict | None:
    url = (
        f"{REDEMET_API_BASE_URL}/mensagens/metar/{icao}"
        f"?api_key={REDEMET_API_KEY}"
        f"&data_ini={date_ini}&data_fim={date_fim}"
        f"&page_tam={PAGE_SIZE}&page={page}"
    )
    try:
        resp = session.get(url, timeout=(10, 120))
    except requests.RequestException as e:
        logging.warning("[%s] Erro de rede: %s", icao, e)
        return None

    if resp.status_code != 200:
        logging.warning("[%s] HTTP %s  Body(<=300c): %.300s", icao, resp.status_code, resp.text)
        return None

    try:
        return resp.json()
    except ValueError:
        logging.warning("[%s] JSON inválido  Body(<=300c): %.300s", icao, resp.text)
        return None


def download_station(
    session: requests.Session,
    icao: str,
    start_date: datetime,
    end_date: datetime,
    out_dir: Path,
):
    out_file = out_dir / f"{icao}.csv"
    if out_file.exists():
        logging.info("[SKIP] %s já existe: %s", icao, out_file)
        return

    all_records = []

    # Itera mês a mês para respeitar limites de taxa da API
    current = start_date.replace(day=1)
    while current <= end_date:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)

        month_end = min(next_month - timedelta(days=1), end_date)

        # Formato YYYYMMDDHH: início do primeiro dia, fim do último dia do mês
        date_ini = current.strftime("%Y%m%d") + "00"
        date_fim = month_end.strftime("%Y%m%d") + "23"

        logging.info("[%s] %s/%s → intervalo %s–%s", icao, current.strftime("%Y"), current.strftime("%m"), date_ini, date_fim)

        page = 1
        month_records = 0
        while True:
            result = fetch_metar_page(session, icao, date_ini, date_fim, page)
            if result is None:
                break

            if not result.get("status"):
                logging.warning(
                    "[%s] API retornou status=false para %s/%s: %s",
                    icao, date_ini, date_fim, result.get("message", ""),
                )
                break

            inner = result.get("data", {})
            records = inner.get("data", [])
            if not records:
                break

            all_records.extend(records)
            month_records += len(records)

            last_page = inner.get("last_page", 1)
            logging.info("[%s] página %d/%d  (+%d registros)", icao, page, last_page, len(records))

            if page >= last_page:
                break
            page += 1
            time.sleep(0.3)

        logging.info("[%s] %s-%s: %d registros", icao, current.strftime("%Y"), current.strftime("%m"), month_records)
        current = next_month
        time.sleep(0.5)

    if not all_records:
        logging.warning("[%s] Nenhum dado baixado — arquivo não criado.", icao)
        return

    df = pd.DataFrame(all_records)
    df.to_csv(out_file, index=False)
    logging.info("[%s] Salvo: %s (%d linhas)", icao, out_file, len(df))


REDEMET_STATION_META = {
    "SBGL": {"name": "Galeão",          "lat": -22.8089, "lng": -43.2436},
    "SBRJ": {"name": "Santos Dumont",   "lat": -22.9099, "lng": -43.1631},
    "SBSC": {"name": "Santa Cruz",      "lat": -22.9324, "lng": -43.7191},
    "SBAF": {"name": "Campo dos Afonsos","lat": -22.8714, "lng": -43.6175},
}


def _parse_metar_record(mens: str):
    """Return (temp, dew, wind_dir, wind_spd) from a METAR string, or NaNs."""
    import re, math
    NaN = float("nan")
    if not isinstance(mens, str):
        return NaN, NaN, NaN, NaN

    # Temperature/dew: e.g. "26/22" or "M02/M05"
    m = re.search(r"\s(M?\d{2})/(M?\d{2})\s", mens)
    if m:
        def _decode(s):
            return -float(s[1:]) if s.startswith("M") else float(s)
        temp = _decode(m.group(1))
        dew  = _decode(m.group(2))
    else:
        temp, dew = NaN, NaN

    # Wind: e.g. "27002KT" or "VRB03KT"
    m2 = re.search(r"\s(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\s", mens)
    if m2 and m2.group(1) != "VRB":
        wind_dir = float(m2.group(1))
        wind_spd = float(m2.group(2)) * 0.514444  # knots → m/s
    else:
        wind_dir, wind_spd = NaN, NaN

    return temp, dew, wind_dir, wind_spd


def _rh_from_td(t, td):
    """Magnus formula relative humidity from temperature and dew point (°C)."""
    import math
    if math.isnan(t) or math.isnan(td):
        return float("nan")
    return 100.0 * math.exp(17.625 * td / (243.04 + td)) / math.exp(17.625 * t / (243.04 + t))


def build_malha1_redemet(out_dir: Path) -> None:
    """Parse all SB*.csv METAR files and write malha1_redemet_diario.csv."""
    import glob as _glob
    import numpy as np

    station_files = sorted(_glob.glob(str(out_dir / "SB*.csv")))
    if not station_files:
        raise FileNotFoundError(f"No SB*.csv files found in {out_dir}")

    daily_frames = []
    for fpath in station_files:
        icao = Path(fpath).stem
        meta = REDEMET_STATION_META.get(icao, {"name": icao, "lat": np.nan, "lng": np.nan})
        logging.info("[malha1] Parsing %s (%s)", icao, fpath)

        df = pd.read_csv(fpath)
        df["validade_inicial"] = pd.to_datetime(df["validade_inicial"], errors="coerce")
        df = df.dropna(subset=["validade_inicial"])

        parsed = df["mens"].apply(lambda s: pd.Series(_parse_metar_record(s),
                                                       index=["TEMP", "DEW", "VD", "VV"]))
        df = pd.concat([df, parsed], axis=1)
        df["RH"] = df.apply(lambda r: _rh_from_td(r["TEMP"], r["DEW"]), axis=1)
        df["DATE"] = df["validade_inicial"].dt.normalize()

        grp = df.groupby("DATE")
        daily = pd.DataFrame({
            "DT_MEDICAO": grp["DATE"].first().index.strftime("%Y-%m-%d"),
            "TEM_AVG":  grp["TEMP"].mean().values,
            "TEM_MIN":  grp["TEMP"].min().values,
            "TEM_MAX":  grp["TEMP"].max().values,
            "DEW_AVG":  grp["DEW"].mean().values,
            "HUM_AVG":  grp["RH"].mean().values,
            "VV_AVG":   grp["VV"].mean().values,
            "VD_AVG":   grp["VD"].mean().values,
            "N_OBS":    grp["TEMP"].count().values,
        })
        daily["STATION"] = icao
        daily["LAT"] = meta["lat"]
        daily["LNG"] = meta["lng"]
        daily["NAME"] = meta["name"]
        daily_frames.append(daily)

    result = pd.concat(daily_frames, ignore_index=True)
    col_order = [
        "DT_MEDICAO", "STATION", "TEM_MIN", "TEM_MAX", "TEM_AVG",
        "DEW_AVG", "HUM_AVG", "VV_AVG", "VD_AVG", "N_OBS", "LAT", "LNG", "NAME",
    ]
    result = result[col_order].sort_values(["STATION", "DT_MEDICAO"]).reset_index(drop=True)
    out_path = out_dir / "malha1_redemet_diario.csv"
    result.to_csv(out_path, index=False)
    logging.info("[malha1] Saved %s (%d rows, %d stations)", out_path, len(result), result["STATION"].nunique())


def main():
    parser = argparse.ArgumentParser(description="Download de dados METAR via REDEMET API")
    parser.add_argument("--malha1-only", action="store_true",
                        help="Apenas parsear CSVs existentes em malha1_redemet_diario.csv")
    parser.add_argument("--city", default=None, help="Código da cidade (ex: RJ)")
    parser.add_argument("--start", default=None, help="Data inicial (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Data final (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="data/raw/redemet", help="Diretório de saída")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.malha1_only:
        build_malha1_redemet(out_dir)
        return

    if not args.city or not args.start or not args.end:
        parser.error("--city, --start, --end são obrigatórios para download")

    city = args.city.upper()
    if city not in STATIONS_BY_CITY:
        raise ValueError(
            f"Cidade '{city}' não configurada. Disponíveis: {list(STATIONS_BY_CITY)}"
        )

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    session = build_session()
    stations = STATIONS_BY_CITY[city]
    logging.info(
        "Iniciando download REDEMET — %d estações para %s (%s → %s)",
        len(stations), city, args.start, args.end,
    )

    for st in stations:
        download_station(session, st["icao"], start_date, end_date, out_dir)

    logging.info("Download REDEMET concluído.")


if __name__ == "__main__":
    main()
