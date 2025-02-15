import pandas as pd
import geopandas as gpd
import pathlib
from pathlib import Path

##### The following code is for the extraction of data from the Himalayan Database
# Documentation to run the Himalayan Database on MacOS can be found here: https://www.himalayandatabase.com/crossover.html
himalayan_data_path = Path('data/himalayas_data')

himalaya_files = {
    "exped": himalayan_data_path / "exped.DBF", # year range is 1905 - 2024
    # "filters":  himalayan_data_path / "filters.DBF",
    "members": himalayan_data_path / "members.DBF",
    # "peaks":  himalayan_data_path / "peaks.DBF",
    # "setup":  himalayan_data_path / "SETUP.DBF"
}

metadata_columns = {}

def extract_metadata(file_path: Path):
    try:
        df = gpd.read_file(file_path) 
        # Following line prints out teh total number of rows in the dataset: 11425
        # print(f"Year length: {len(df['YEAR'])}\nHOST length: {len(df['HOST'])}\nSeason Length: {len(df['SEASON'])}")
        return list(df.columns)  
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None 

for name, file_path in himalaya_files.items():
    metadata_columns[name] = extract_metadata(file_path)

# Printing loop
# for file, columns in metadata_columns.items():
#     print(f"\nMetadata for {file}:")
#     if columns:
#         print(columns)
#     else:
#         print("Could not read columns.")

##### The following code is for the extraction of data from the EU's Copernicus Climate Data Store "ERA5 hourly data on single levels from 1940 to present" dataset
import cdsapi
import os
import concurrent.futures

# Buffered coordinates = Exact coordinates +- 1.5° to incorporate relevant surrounding meteorological data
mountain_areas = {
    "Everest": [29.29, 85.25, 26.29, 88.25],
    "K2": [37.22, 75.00, 34.22, 78.00],
    "Kangchenjunga": [29.12, 86.38, 26.12, 89.38],
    "Lhotse": [29.27, 85.25, 26.27, 88.25],
    "Makalu": [29.23, 85.35, 26.23, 88.35],
    "Cho Oyu": [29.35, 85.09, 26.35, 88.09],
    "Dhaulagiri I": [30.11, 81.59, 27.11, 84.59],
    "Manaslu": [30.03, 83.03, 27.03, 86.03],
    "Nanga Parbat": [36.44, 73.05, 33.44, 76.05],
    "Annapurna I": [30.05, 82.19, 27.05, 85.19],
    "Gasherbrum I": [37.13, 75.11, 34.13, 78.11],
    "Broad Peak": [37.18, 75.04, 34.18, 78.04],
    "Gasherbrum II": [37.15, 75.09, 34.15, 78.09],
    "Shishapangma": [29.51, 84.16, 26.51, 87.16],
}

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
    # "year": [
    #     "1940", "1941", "1942",
    #     "1943", "1944", "1945",
    #     "1946", "1947", "1948",
    #     "1949", "1950", "1951",
    #     "1952", "1953", "1954",
    #     "1955", "1956", "1957",
    #     "1958", "1959", "1960",
    #     "1961", "1962", "1963",
    #     "1964", "1965", "1966",
    #     "1967", "1968", "1969",
    #     "1970", "1971", "1972",
    #     "1973", "1974", "1975",
    #     "1976", "1977", "1978",
    #     "1979", "1980", "1981",
    #     "1982", "1983", "1984",
    #     "1985", "1986", "1987",
    #     "1988", "1989", "1990",
    #     "1991", "1992", "1993",
    #     "1994", "1995", "1996",
    #     "1997", "1998", "1999",
    #     "2000", "2001", "2002",
    #     "2003", "2004", "2005",
    #     "2006", "2007", "2008",
    #     "2009", "2010", "2011",
    #     "2012", "2013", "2014",
    #     "2015", "2016", "2017",
    #     "2018", "2019", "2020",
    #     "2021", "2022", "2023",
    #     "2024"
    # ],
    "year": [],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
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

client = cdsapi.Client()
era5_dataset = "reanalysis-era5-single-levels"
era5_data_path = Path('data/era5_data')
era5_data_path.mkdir(parents=True, exist_ok=True)
total_size = 0

# 5-year consecutive batches due to API request limits
year_batches = []
for year in range(2024, 1940, -4):
    batch = [year, year - 1, year - 2, year - 3]
    year_batches.append(batch)

year_batches.append([1940]) # we have 85 values where 85 % 4 = 1, so the last value will be in its own batch
# print(year_batches)

stop_requests = False

def download_batch (mountain, area, year_batch, mountain_folder):
    """
    Function to request ERA5 data in parallel since the download time depends on the API processing the request.
    """

    global stop_requests
    if stop_requests:
        return 0

    batch_start, batch_end = year_batch[0], year_batch[-1]
    file_path = mountain_folder / f"{mountain}_{batch_start}_{batch_end}.grib"

    if file_path.exists():
        print(f"Skipping {mountain}_{batch_start}_{batch_end}. Already downloaded.")
        return file_path.stat().st_size

    request = base_request.copy()
    request["year"] = [str(y) for y in year_batch]
    request["area"] = area
    print(f"Requesting {mountain} for {batch_start}-{batch_end}...")

    try:

        client.retrieve(era5_dataset, request, str(file_path))
        file_size = file_path.stat().st_size if file_path.exists() else 0
        return file_size
    
    except Exception as e:

        error_message = str(e)

        if "Number of API queued requests for this dataset is temporally limited." in error_message:
            print("API limit reached. Stopping all new requests.")
            stop_requests = True
            return None
        
        print(f"Error retrieving data for {mountain} {batch_start}-{batch_end}: {e}")
        return 0

# Documentation to run the request: https://cds.climate.copernicus.eu/how-to-api
# We have 22 requests per mountain since 85 % 4 = 1. We have 14 mountains, so 294 requests in total
with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
    batch_to_request = {}

    for mountain, area in mountain_areas.items():

        print(f"Downloading data for {mountain}...")
        mountain_folder = era5_data_path / mountain
        mountain_folder.mkdir(parents=True, exist_ok=True)

        for year_batch in year_batches:
            next_batch = executor.submit(download_batch, mountain, area, year_batch, mountain_folder)
            batch_to_request[next_batch] = (mountain, year_batch)
    
    for next_batch in concurrent.futures.as_completed(batch_to_request):
        mountain, year_batch = batch_to_request[next_batch]

        if stop_requests:
            print("Stopping requests due to API limit.")
            executor.shutdown(wait=False)
            exit()
        
        try:
            size = next_batch.result()
            total_size += size
            print(f"Completed {mountain} {year_batch[0]}-{year_batch[-1]}. File size: {size / (1024 * 1024):.2f} MB")

        except Exception as e:
            print(f"Download failed for {mountain} {year_batch[0]}-{year_batch[-1]}: {e}")

total_size_mb = total_size / (1024 * 1024)
total_size_gb = total_size_mb / 1024
print("All downloads completed successfully! Total file size; {total_size_gb:.2f} GB")
