import ftplib
import logging
import datetime
import os
import argparse

def download_cnes_file(filetype, uf, date, destination):
    """
    Downloads a file from SUS FTP server to a local destination.

    Args:
    filetype: CNES file type
    uf: str, UF in ISO 3166-2 format
    date: str, Date in yymm format
    path: str, Destination path

    """    
    try:
        ftp_url = f"ftp://ftp.datasus.gov.br/dissemin/publicos/CNES/200508_/Dados/{filetype}/{filetype}{uf}{date}.dbc"
        logging.info(f"Retrieving file: {ftp_url}")

        directory = os.path.dirname(destination)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")
                    
        parsed_url = ftp_url.replace("ftp://", "").split("/")
        ftp_host = parsed_url[0]
        ftp_path = "/".join(parsed_url[1:-1])
        file_name = parsed_url[-1]

        ftp = ftplib.FTP(ftp_host)
        ftp.login()  # login as anonymous
        ftp.cwd(ftp_path)

        with open(destination, "wb") as file:
            ftp.retrbinary("RETR " + file_name, file.write)

        ftp.quit()
        logging.info(f"File downloaded successfully at: {destination}")
    except ftplib.all_errors as e:
        logging.error(f"FTP error: {e}")
        raise        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download CNES Estabelecimentos file from SUS")
    parser.add_argument("filetype", default="ST", help="CNES file type, default is ST")
    parser.add_argument("uf", default="RJ", help="UF in ISO 3166-2 format, default is RJ")
    parser.add_argument("date", help="Date in yymm format, default is current year/month", default=None)
    parser.add_argument("path", help="Destination path", default="data/raw/cnes/STRJ2311.dbc")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    if args.date is None:
        args.date =  f'{datetime.date.today().strftime("%y")}{datetime.date.today().strftime("%m")}'

    download_cnes_file(args.filetype, args.uf, args.date, args.path)

if __name__ == "__main__":
    main()