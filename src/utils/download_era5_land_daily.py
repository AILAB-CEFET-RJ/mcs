import calendar
import logging
import datetime
import os
import argparse
import cdsapi

def generate_days(month, year):
    num_days = calendar.monthrange(year, month)[1]
    days_as_strings = list(range(1, num_days + 1))
    return days_as_strings

def download_era5_land_daily(variable, date, daily_statistic, destination):
    try:
        if len(date) != 6 or not date[:4].isdigit() or not (1 <= int(date[4:]) <= 12):
            raise ValueError("Invalid date format. Use 'yyyymm'.")

        year = date[:4]
        month = date[4:6]
        days = generate_days(int(month), int(year))

        directory = os.path.dirname(destination)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")


        dataset = "derived-era5-land-daily-statistics"
        request = {
            "variable": [variable],
            "year": year,
            "month": month,
            "day": days,
            "daily_statistic": "daily_mean",
            "time_zone": "utc-03:00",
            "frequency": "1_hourly"
        }


        client = cdsapi.Client()
        client.retrieve(dataset, request).download(destination)
        logging.info(f"File downloaded successfully at: {destination}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download ERA 5 data from Copernicus")
    parser.add_argument("--variable", default="2m_temperature", help="Variable from ERA5 dataset")
    parser.add_argument("--date", help="Date in yyyymm format", default=None, required=False)
    parser.add_argument("--daily_statistic", choices=["daily_mean", "daily_minimum", "daily_maximum"], required=True)
    parser.add_argument("--path", help="Destination path", default="data/raw/era5_land_daily", required=False)
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), 
                        format="%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s")

    if args.date is None:
        args.date = datetime.date.today().strftime("%Y%m")

    download_era5_land_daily(args.variable, args.date, args.daily_statistic, args.path)

if __name__ == "__main__":
    main()
    #download_era5_land_daily("skin_temperature", "202402", "daily_mean", "data/raw/era5_land_daily")