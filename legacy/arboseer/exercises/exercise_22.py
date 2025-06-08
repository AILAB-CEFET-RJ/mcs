import xarray as xr
import pandas as pd

# Define the NetCDF file path
file_path = r"data\raw\era5\RJ_1997_2024.nc"
output_txt = "netcdf_summary.txt"

# Try loading the dataset
try:
    data = xr.open_dataset(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Prepare a string to collect all the information
summary_info = []

# Add dataset summary
summary_info.append("=== Dataset Summary ===")
summary_info.append(str(data))

# Add dimensions
summary_info.append("\n=== Dimensions ===")
summary_info.append(str(data.dims))

# Add coordinates
summary_info.append("\n=== Coordinates ===")
summary_info.append(str(list(data.coords.keys())))

# Add variables
summary_info.append("\n=== Variables (Products) ===")
variables = list(data.data_vars.keys())
summary_info.append(str(variables))

# Collect detailed metadata for each variable
summary_info.append("\n=== Variable Details ===")
for var in variables:
    attrs = data[var].attrs
    summary_info.append(f"Variable: {var}")
    summary_info.append(f"  Description: {attrs.get('long_name', 'N/A')}")
    summary_info.append(f"  Units: {attrs.get('units', 'N/A')}")
    summary_info.append(f"  Statistic: {attrs.get('statistic', 'N/A')}")
    summary_info.append(f"  Other Attributes: {attrs}")
    summary_info.append("\n")

# Write all the information to a text file
with open(output_txt, "w") as txt_file:
    txt_file.write("\n".join(summary_info))

# Inform the user
print(f"Summary saved to: {output_txt}")

# Close the dataset
data.close()
