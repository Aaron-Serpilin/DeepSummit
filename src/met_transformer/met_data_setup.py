import xarray as xr
import pygrib
import glob
import pandas as pd
from pathlib import Path

weather_variable_list = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "surface_pressure",
    "total_precipitation",
    "ice_temperature_layer_1",
    "ice_temperature_layer_2",
    "ice_temperature_layer_3",
    "ice_temperature_layer_4",
    "maximum_2m_temperature_since_previous_post_processing",
    "minimum_2m_temperature_since_previous_post_processing",
    "skin_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "10m_u_component_of_neutral_wind",
    "10m_v_component_of_neutral_wind",
    "10m_wind_gust_since_previous_post_processing",
    "instantaneous_10m_wind_gust",
    "cloud_base_height",
    "high_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "total_cloud_cover",
    "total_column_cloud_ice_water",
    "total_column_cloud_liquid_water",
    "vertical_integral_of_divergence_of_cloud_frozen_water_flux",
    "vertical_integral_of_divergence_of_cloud_liquid_water_flux",
    "vertical_integral_of_eastward_cloud_frozen_water_flux",
    "vertical_integral_of_eastward_cloud_liquid_water_flux",
    "vertical_integral_of_northward_cloud_frozen_water_flux",
    "vertical_integral_of_northward_cloud_liquid_water_flux",
    "convective_precipitation",
    "convective_rain_rate",
    "instantaneous_large_scale_surface_precipitation_fraction",
    "large_scale_rain_rate",
    "large_scale_precipitation",
    "large_scale_precipitation_fraction",
    "maximum_total_precipitation_rate_since_previous_post_processing",
    "minimum_total_precipitation_rate_since_previous_post_processing",
    "precipitation_type",
    "total_column_rain_water",
    "convective_snowfall",
    "convective_snowfall_rate_water_equivalent",
    "large_scale_snowfall_rate_water_equivalent",
    "large_scale_snowfall",
    "snow_albedo",
    "snow_density",
    "snow_depth",
    "snow_evaporation",
    "snowfall",
    "snowmelt",
    "temperature_of_snow_layer",
    "total_column_snow_water",
]

mountain_areas = {
    "Everest": ["29.29", "85.25", "26.29", "88.25"]
    # "Kangchenjunga": ["29.12", "86.38", "26.12", "89.38"],
    # "Lhotse": ["29.27", "85.25", "26.27", "88.25"],
    # "Makalu": ["29.23", "85.35", "26.23", "88.35"],
    # "Cho Oyu": ["29.35", "85.09", "26.35", "88.09"],
    # "Dhaulagiri I": ["30.11", "81.59", "27.11", "84.59"],
    # "Manaslu": ["30.03", "83.03", "27.03", "86.03"],
    # "Annapurna I": ["30.05", "82.19", "27.05", "85.19"],
}

# Convert the era5_data files to grib for data processing
def file_to_grib (home_path: Path):
    for mountain_dir in home_path.iterdir():
        if not mountain_dir.is_dir():
            continue
        
        for file in mountain_dir.iterdir():
            if not file.is_file() or file.name.startswith("."):
              continue
              
            new_path = file.with_suffix(".grib")
            print(f"Renaming: {file} -> {new_path}\n")
            file.rename(new_path)

# The weather_variable_list are not the variables pygrib uses internally
def get_variable_mapping (grib_path: Path):
  grbs = pygrib.open(str(grib_path))
  mapping = {msg.name: msg.shortName for msg in grbs}
  grbs.close()
  return mapping

# We transform the pygrib file into a dataframe by grouping based ont he date, name, and value of the dataset
def process_pygrib( grib_path: Path,
                              vars_to_keep: list[str]
                              ) -> pd.DataFrame:
    
  grbs = pygrib.open(str(grib_path))  
  records = []
  for msg in grbs:
     if msg.shortName in vars_to_keep:
        records.append({
           'time': msg.validDate,
           'variable': msg.shortName,
           'value': float(msg.values.mean())
        })

  grbs.close()

  if not records:
     raise ValueError(f"No records found in {grib_path}.")
  
  df = pd.DataFrame(records)
  # Each shortName (variable) is a column
  df = df.pivot_table(index='time', columns='variable', values='value')
  df = df.sort_index()
  return df

def process_grib_to_csv (era5_root: Path,
                         csv_root: Path,
                         vars_to_keep: list[str],
                         ) -> None:
   
   for mountain_dir in era5_root.iterdir():
      if not mountain_dir.is_dir():
         continue
      out_dir = csv_root / mountain_dir.name
      out_dir.mkdir(parents=True, exist_ok=True)

      for grib_file in mountain_dir.glob('*.grib'):
          try:
            df = process_pygrib(grib_file, vars_to_keep)
            out_csv = out_dir / f"{grib_file.stem}.csv"
            df.to_csv(out_csv)
            print(f"Saved: {out_csv}")
          except ValueError as e:
            print(f"Skipping {grib_file}: {e}")

era5_path = Path('data/era5_data/database_files')
output_path = Path('data/era5_data/processed_csvs')
output_path.mkdir(exist_ok=True, parents=True)

mountain = "Everest"
file = "Everest-1940-1944.grib"
mountain_path = era5_path / mountain / file
csv_path = era5_path / "processed_csvs"

sample = next(era5_path.iterdir()) / (next(era5_path.iterdir()).glob("*.grib").__next__().name)
mapping = get_variable_mapping(sample)

# for long_name, short in mapping.items():
#       print(f"{short:8} ← {long_name}")

process_grib_to_csv(era5_path, csv_path, mapping)

