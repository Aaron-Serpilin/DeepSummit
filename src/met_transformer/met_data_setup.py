import xarray as xr
import glob
import pandas as pd
from pathlib import Path

era5_path = Path('data/era5_data/database_files')
output_path = Path('data/era5_data/processed_csvs')
output_path.mkdir(exist_ok=True, parents=True)

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

file_to_grib(era5_path)


