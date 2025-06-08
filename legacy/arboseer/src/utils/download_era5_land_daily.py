import os
import cdsapi

def download_era5_land_daily_batch(variable, statistics, years, extent, output_dir):
    """
    Downloads ERA5-Land daily data for a given variable and statistics as yearly batches.
    
    Args:
        variable (str): The variable to download (e.g., "skin_temperature").
        statistics (list): List of daily statistics to download (e.g., ["daily_mean", "daily_minimum", "daily_maximum"]).
        years (list): List of years to download (e.g., [2019, 2020, 2021, 2022, 2023]).
        extent (list): Geographic extent in the format [north, west, south, east].
        output_dir (str): Directory where the downloaded files will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the CDS API client
    client = cdsapi.Client()

    # Loop over years and statistics
    for year in years:
        for statistic in statistics:
            output_file = os.path.join(output_dir, f"{variable}_{statistic}_{year}.nc")

            # Skip if the file already exists
            if os.path.exists(output_file):
                print(f"File already exists, skipping: {output_file}")
                continue

            # Define the request parameters
            request = {
                "variable": variable,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],  # All months
                "day": [f"{d:02d}" for d in range(1, 32)],   # All days
                "time_zone": "utc-03:00",
                "daily_statistic": statistic,
                "format": "netcdf",
                "area": extent,  # [north, west, south, east]
            }

            try:
                print(f"Requesting data for {year}, {statistic}: {output_file}")
                client.retrieve("derived-era5-land-daily-statistics", request).download(output_file)
                print(f"Downloaded: {output_file}")
            except Exception as e:
                print(f"Failed to download {output_file}: {e}")

if __name__ == "__main__":
    # Define parameters
    variable = "skin_temperature"
    statistics = ["daily_mean", "daily_minimum", "daily_maximum"]
    years = [2019, 2020, 2021, 2022, 2023]
    extent = [-20.7634, -44.7930, -23.3702, -40.7635]  # [north, west, south, east]
    output_dir = "data/era5_land_daily"

    # Call the function
    download_era5_land_daily_batch(variable, statistics, years, extent, output_dir)