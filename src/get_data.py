import pandas as pd
import geopandas as gpd
import pathlib
from pathlib import Path

# Documentation to run the Himalayan Database on MacOS can be found here: https://www.himalayandatabase.com/crossover.html
data_path = Path('data/himalayas_data')

dbf_files = {
    "exped": data_path / "exped.DBF", # year range is 1905 - 2024
    # "filters": data_path / "filters.DBF",
    "members": data_path / "members.DBF",
    # "peaks": data_path / "peaks.DBF",
    # "setup": data_path / "SETUP.DBF"
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

for name, file_path in dbf_files.items():
    metadata_columns[name] = extract_metadata(file_path)

# Printing loop
for file, columns in metadata_columns.items():
    print(f"\nMetadata for {file}:")
    if columns:
        print(columns)
    else:
        print("Could not read columns.")