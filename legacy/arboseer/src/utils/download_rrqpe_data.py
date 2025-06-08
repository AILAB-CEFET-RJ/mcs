from utils.download_goes_prod import download_goes_prod
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_rrqpe_data(date, path):
    product = "ABI-L2-RRQPEF"
    os.makedirs(path, exist_ok=True)  # Ensure output directory exists

    def download_for_time(yyyymmddhhmn):
        download_goes_prod(yyyymmddhhmn, product, path, 'hour')
    
    
    futures = []
    with ThreadPoolExecutor() as executor:

        for hour in np.arange(0,24-3,1):
            yyyymmddhhmn = f"{date}{hour:02.0f}00"
            futures.append(executor.submit(download_for_time, yyyymmddhhmn))

        for hour in np.arange(24-3,24,1):       
            previous_day = datetime.strptime(date, '%Y%m%d')
            previous_day -= timedelta(days=1)
            yyyymmddhhmn = f"{previous_day.strftime('%Y%m%d')}{hour:02.0f}00"
            futures.append(executor.submit(download_for_time, yyyymmddhhmn))
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()  # This will wait for each task to finish

def main():
    parser = argparse.ArgumentParser(description="Download ABI-L2-RRQPE product from GOES-16 for a date")
    parser.add_argument("date", help="Date in yyyymmdd format")
    parser.add_argument("path", help="Destination path, default is data/raw", default="data/raw/rrqpe")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="DEBUG", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    download_rrqpe_data(args.date, args.path)

if __name__ == "__main__":
    main()
