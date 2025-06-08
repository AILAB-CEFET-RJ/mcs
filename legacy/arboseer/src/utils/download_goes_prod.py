import boto3                             
from botocore import UNSIGNED            
from botocore.config import Config       
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_file(s3_client, bucket_name, key, local_file_path):
    if not os.path.exists(local_file_path):
        try:
            s3_client.download_file(bucket_name, key, local_file_path)
            logging.info(f'Downloaded file {local_file_path}')
        except Exception as e:
            logging.error(f'Error downloading {key}: {e}')
    else:
        logging.info(f'File {local_file_path} already exists')

def download_goes_prod(date, product_name, path_dest, granularity='minute'):
    """
    Download product from AMAZON AWS GOES-16 repo

    Args:
    date: str, date
    product_name: str, product name
    path_dest: str, Destination path
    granularity: str, granularity of the download ('minute', 'hour', 'day', 'year')

    """      
    os.makedirs(path_dest, exist_ok=True)

    year = datetime.strptime(date, '%Y%m%d%H%M').strftime('%Y')
    day_of_year = datetime.strptime(date, '%Y%m%d%H%M').strftime('%j')
    hour = datetime.strptime(date, '%Y%m%d%H%M').strftime('%H')
    minute = datetime.strptime(date, '%Y%m%d%H%M').strftime('%M')

    # AMAZON repository information
    bucket_name = 'noaa-goes16'

    # Initializes the S3 client
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Define the prefix based on the granularity
    if granularity == 'minute':
        prefix = f'{product_name}/{year}/{day_of_year}/{hour}/OR_{product_name}-M6_G16_s{year}{day_of_year}{hour}{minute}'
    elif granularity == 'hour':
        prefix = f'{product_name}/{year}/{day_of_year}/{hour}/'
    elif granularity == 'day':
        prefix = f'{product_name}/{year}/{day_of_year}/'
    elif granularity == 'year':
        prefix = f'{product_name}/{year}/'
    else:
        logging.error(f'Invalid granularity: {granularity}')
        return -1

    # Search for files on the server
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    files_downloaded = 0
    futures = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    file_name = key.split('/')[-1]
                    local_file_path = os.path.join(path_dest, file_name)
                    
                    futures.append(executor.submit(download_file, s3_client, bucket_name, key, local_file_path))

        for future in as_completed(futures):
            try:
                future.result()
                files_downloaded += 1
            except Exception as e:
                logging.error(f'Error in downloading file: {e}')

    if files_downloaded == 0:
        logging.error(f'No files found for the date: {date}, Product-{product_name}, Granularity-{granularity}')
        return -1
    else:
        return files_downloaded

def main():
    parser = argparse.ArgumentParser(description="Download GOES-16 product")
    parser.add_argument("date", help="File date in yyyymmddhhmn format")
    parser.add_argument("product_name", help="GOES-16 product, default is ABI-L2-LSTF", default="ABI-L2-LSTF")
    parser.add_argument("path_dest", help="Destination path, default is data/raw/goes", default="data/raw/goes")
    parser.add_argument("granularity", help="Granularity, default is hour", default="hour")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="DEBUG", help="Set the logging level")
    
    args = parser.parse_args()
        
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    download_goes_prod(args.date, args.product_name, args.path_dest, args.granularity)

if __name__ == "__main__":
    main()
