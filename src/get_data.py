import pandas as pd
import geopandas as gpd
import pathlib
from pathlib import Path

##### The following code is for the extraction of data from the EU's Copernicus Climate Data Store "ERA5 hourly data on single levels from 1940 to present" dataset
import cdsapi
import os
import concurrent.futures

base_request = {
    "product_type": ["reanalysis"],
    "variable": [
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
        "total_column_snow_water"
    ],
    "year": [], # We leave this empty since we have to customize each batch due to the API limit
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"], 
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
        ],
    "time": ["00:00"],
    "data_format": "grib",
    "download_format": "zip",
    "area": []
}

# Buffered coordinates = Exact coordinates +- 1.5° to incorporate relevant surrounding meteorological data
mountain_areas = {
    "Everest": ["29.29", "85.25", "26.29", "88.25"],
    "Kangchenjunga": ["29.12", "86.38", "26.12", "89.38"],
    "Lhotse": ["29.27", "85.25", "26.27", "88.25"],
    "Makalu": ["29.23", "85.35", "26.23", "88.35"],
    "Cho Oyu": ["29.35", "85.09", "26.35", "88.09"],
    "Dhaulagiri I": ["30.11", "81.59", "27.11", "84.59"],
    "Manaslu": ["30.03", "83.03", "27.03", "86.03"],
    "Annapurna I": ["30.05", "82.19", "27.05", "85.19"],
    # "Nanga Parbat": ["36.44", "73.05", "33.44", "76.05"],
    # "K2": ["37.22", "75.00", "34.22", "78.00"],
    # "Gasherbrum I": ["37.13", "75.11", "34.13", "78.11"],
    # "Broad Peak": ["37.18", "75.04", "34.18", "78.04"],
    # "Gasherbrum II": ["37.15", "75.09", "34.15", "78.09"],
    # "Shishapangma": ["29.51", "84.16", "26.51", "87.16"],
}

client = cdsapi.Client()
era5_dataset = "reanalysis-era5-single-levels"
era5_data_path = Path('data/era5_data')
era5_data_path.mkdir(parents=True, exist_ok=True)

# Here we create the 5-year batches for the data
year_batches = []
current_end = 2024
while current_end >= 1940:
    current_start = max(1940, current_end - 4)
    batch = list(range(current_start, current_end + 1))
    year_batches.append(batch)
    current_end = current_start - 1

def submit_request(mountain, year_batch):
    """
    For a given mountain and a 5-year batch, checks if the file exists.
    If not, submits a data request to the ERA5 API.
    The file is saved as {mountain}-{start_year}-{end_year}.zip.
    """
    start_year, end_year = year_batch[0], year_batch[-1]
    tokenized_mountain = mountain.replace(" ", "-")
    mountain_folder = era5_data_path / mountain
    mountain_folder.mkdir(parents=True, exist_ok=True)
    output_file = mountain_folder / f"{tokenized_mountain}-{start_year}-{end_year}"
    
    if output_file.exists():
        print(f"File {output_file} already exists. Skipping request for {mountain} {start_year}-{end_year}.")
        return

    request = base_request.copy()
    request["year"] = [str(y) for y in year_batch]
    request["area"] = mountain_areas[mountain]

    print(f"Submitting request for {mountain} for years {start_year}-{end_year}...")
    try:
        client.retrieve(era5_dataset, request, str(output_file))
        # print(f"Request submitted for {mountain} for years {start_year}-{end_year}.")
    except Exception as e:
        print(f"Error submitting request for {mountain} for years {start_year}-{end_year}: {e}")

def request_mountain_data(mountain):
    """
    Submits a given number of concurrent requests (one for each 5-year batch from 2024 to 1940)
    for the given mountain.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=17) as executor:
        future_requests = {
            executor.submit(submit_request, mountain, year_batch): year_batch
            for year_batch in year_batches
        }
        for future in concurrent.futures.as_completed(future_requests):
            year_batch = future_requests[future]
            start_year, end_year = year_batch[0], year_batch[-1]
            try:
                future.result()
                # print(f"Submission completed for {mountain} for years {start_year}-{end_year}.")
            except Exception as e:
                print(f"Submission failed for {mountain} for years {start_year}-{end_year}: {e}")

request_mountain_data("Kangchenjunga")
