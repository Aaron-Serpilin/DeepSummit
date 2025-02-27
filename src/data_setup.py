import pygrib
from pathlib import Path

era5_data_path = Path("data/era5_data/Everest/Everest-2020-2024")

# Open the GRIB file using pygrib
with pygrib.open(era5_data_path) as grbs:
    for grb in grbs:
        # Print a summary of header information for each message
        print(grb)


