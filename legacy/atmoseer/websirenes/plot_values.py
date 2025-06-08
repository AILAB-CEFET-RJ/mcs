from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_tp_values(
    websirenes_df: pd.DataFrame,
    era5land_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    station_name: str,
):
    plt.figure(figsize=(10, 5))
    plt.plot(era5land_df.index, era5land_df["tp"], label="ERA5Land")
    plt.plot(websirenes_df.index, websirenes_df["m15"], label="WebSirenes")

    plt.title("Total Precipitation values")
    plt.xlabel("Date")
    plt.ylabel("Total Precipitation")

    plt.legend()

    plt.savefig(
        Path(__file__).parent
        / f"Total_Precipitation_values_{station_name}_{start_date}_{end_date}.png"
    )

    plt.show()
