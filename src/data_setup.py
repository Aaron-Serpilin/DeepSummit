import xarray as xr

grib_file = "data/era5_data/January/Himalayas_January_1982_2024.grib"

ds = xr.open_dataset(grib_file, engine="cfgrib")

# Print dataset summary
print(ds)
