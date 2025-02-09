import pandas as pd
import sys
from datetime import datetime
import logging
import os
import requests

INMET_API_BASE_URL = "https://apitempo.inmet.gov.br"

INMET_WEATHER_STATION_IDS = (
    'A617',
    'A615',
    'A628',
    'A606',
    'A502',
    'A604',
    'A607',
    'A620',
    'A629',
    'A557',
    'A603',
    'A518',
    'A608',
    'A517',
    'A627',
    'A624',
    'A570',
    'A513',
    'A619',
    'A529',
    'A610',
    'A622',
    'A637',
    'A609',
    'A626',
    'A652',
    'A636',
    'A621',
    'A602',
    'A630',
    'A514',
    'A667',
    'A601',
    'A659',
    'A618',
    'A625',
    'A611',
    'A633',
    'A510',
    'A634',
    'A612'
)

def retrieve_from_station(station_id, beginning_year, end_year, api_token, output_path):

    os.makedirs(output_path, exist_ok=True)

    current_date = datetime.now()
    current_year = current_date.year
    end_year = min(end_year, current_year)

    years = list(range(beginning_year, end_year + 1))
    df_inmet_stations = pd.read_json(INMET_API_BASE_URL + '/estacoes/T')
    station_row = df_inmet_stations[df_inmet_stations['CD_ESTACAO'] == station_id]
    df_observations_for_all_years = None
    logging.info(f"Downloading observations from weather station {station_id}...")
    for year in years:
        logging.info(f"Downloading observations for year {year}...")

        if year == end_year:
            datetime_str = str(end_year) + '-12-31'
            last_date_of_the_end_year = datetime.strptime(datetime_str, '%Y-%m-%d')
            end_date = min(last_date_of_the_end_year, current_date)
            end_date_str = end_date.strftime("%Y-%m-%d")
            query_str = INMET_API_BASE_URL + '/token/estacao/' + str(year) + '-01-01/' + end_date_str + '/' + station_id + "/" + api_token
        else:
            query_str = INMET_API_BASE_URL + '/token/estacao/' + str(year) + '-01-01/' + str(year) + '-12-31/' + station_id + "/" + api_token

        df_observations_for_a_year = []
        response = requests.get(query_str)
        if response.status_code == 200:
            data = response.json()
            df_observations_for_a_year = pd.DataFrame(data)

        if df_observations_for_a_year is None:
            logging.info(f"No observations for year '{year}'.")    

        if df_observations_for_all_years is None:
            temp = [df_observations_for_a_year]
        else:
            temp = [df_observations_for_all_years, df_observations_for_a_year]
        df_observations_for_all_years = pd.concat(temp)

    filename = output_path + '/' + station_row['CD_ESTACAO'].iloc[0] + '.parquet'
    logging.info(f"Done! Saving dowloaded content to '{filename}'.")
    df_observations_for_all_years.to_parquet(filename)

def retrieve_data(station_id, initial_year, final_year, api_token, output_path):
    if station_id == "all":
        df_inmet_stations = pd.read_json(INMET_API_BASE_URL + '/estacoes/T')
        station_row = df_inmet_stations[df_inmet_stations['CD_ESTACAO'].isin(INMET_WEATHER_STATION_IDS)]
        for j in list(range(0, len(station_row))):
            station_id = station_row['CD_ESTACAO'].iloc[j]
            retrieve_from_station(station_id, initial_year, final_year, api_token, output_path)
    else:
        retrieve_from_station(station_id, initial_year, final_year, api_token, output_path)

import argparse

def main(argv):
    station_id = ""

    parser = argparse.ArgumentParser(prog=argv[0], 
                                     usage='{0} -s <ws_id> -b <begin_year> -e <end_year> -t <api_token>'.format(argv[0]), 
                                     description="""This script provides a simple interface for retrieving observations
                                       made by a user-provided weather station from the INMET archive.""")
    parser.add_argument("-t", "--api_token", required=True, help="INMET API token", metavar='')
    parser.add_argument("-s", "--ws_id", required=True, help="Weather station ID", metavar='')
    parser.add_argument("-b", "--begin_year", type=int, required=True, help="Start year", metavar='')
    parser.add_argument("-e", "--end_year", type=int, required=True, help="End year", metavar='')
    parser.add_argument("-o", "--output_path", required=True, help="Output path", metavar='')    

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")   

    args = parser.parse_args(argv[1:])

    api_token = args.api_token
    station_id = args.ws_id
    start_year = args.begin_year
    end_year = args.end_year
    output_path = args.output_path

    if not (station_id in INMET_WEATHER_STATION_IDS):
        parser.error(f'Invalid station ID: {station_id}')

    assert (api_token is not None) and (api_token != '')
    assert (station_id is not None) and (station_id != '')
    assert (start_year <= end_year)

    retrieve_data(station_id, start_year, end_year, api_token, output_path)

if __name__ == "__main__":
    main(sys.argv)