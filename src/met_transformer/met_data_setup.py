import xarray as xr
from pathlib import Path

era5_path = Path('data/era5_data/database_files')
mountain = "Everest"
folder = era5_path / mountain

grib_files = [
    file for file in folder.iterdir()
    if file.is_file() and file.name != '.DS_Store' 
]

ds = xr.open_mfdataset(
    grib_files,
    engine="cfgrib",
    concat_dim="time",
    combine="by_coords",
    preprocess=lambda x: x.sel(
        longitude=slice(85.25, 88.25),
        latitude =slice(26.29, 29.29)
    )
)

print(ds)