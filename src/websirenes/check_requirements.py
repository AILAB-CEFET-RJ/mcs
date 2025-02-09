from pathlib import Path


def check_requirements():
    websirenes_defesa_civil_path = Path(__file__).parent / "websirenes_defesa_civil"
    if not websirenes_defesa_civil_path.exists():
        print(
            f"Expected websirenes_defesa_civil folder in {websirenes_defesa_civil_path}"
        )
        exit(1)

    if not list(websirenes_defesa_civil_path.glob("*.txt")):
        print(
            f"websirenes_defesa_civil folder does not contain any txt files at {websirenes_defesa_civil_path}"
        )
        exit(1)

    websirenes_coords_path = Path(__file__).parent / "websirenes_coords.parquet"
    if not websirenes_coords_path.exists():
        print(f"Expected websirenes_coords.parquet file in {websirenes_coords_path}")
        exit(1)

    era5_land_path = Path(__file__).parent / "ERA5Land"
    if not era5_land_path.exists():
        print(f"Expected ERA5Land folder in {era5_land_path}")
        exit(1)

    monthly_data_path = era5_land_path / "monthly_data"
    montly_data_path = era5_land_path / "montly_data"
    if not monthly_data_path.exists() and not montly_data_path.exists():
        print(f"Expected monthly_data folder in {monthly_data_path}")
        exit(1)

    if not list(monthly_data_path.glob("*.nc")) and not list(
        montly_data_path.glob("*.nc")
    ):
        print(
            f"monthly_data folder does not contain any .nc files at {monthly_data_path}"
        )
        exit(1)
