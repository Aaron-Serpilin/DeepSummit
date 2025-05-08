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

# For convenience instead of running get_variable_mapping each time which takes ~3 minutes
weather_mapping = {
    '10u':   '10 metre U wind component',
    '10v':   '10 metre V wind component',
    '2d':    '2 metre dewpoint temperature',
    '2t':    '2 metre temperature',
    'msl':   'Mean sea level pressure',
    'sst':   'Sea surface temperature',
    'sp':    'Surface pressure',
    'tp':    'Total precipitation',
    'istl1': 'Ice temperature layer 1',
    'istl2': 'Ice temperature layer 2',
    'istl3': 'Ice temperature layer 3',
    'istl4': 'Ice temperature layer 4',
    'mx2t':  'Maximum temperature at 2 metres since previous post-processing',
    'mn2t':  'Minimum temperature at 2 metres since previous post-processing',
    'skt':   'Skin temperature',
    '100u':  '100 metre U wind component',
    '100v':  '100 metre V wind component',
    'u10n':  '10 metre u-component of neutral wind',
    'v10n':  '10 metre v-component of neutral wind',
    '10fg':  'Maximum 10 metre wind gust since previous post-processing',
    'i10fg': 'Instantaneous 10 metre wind gust',
    'cbh':   'Cloud base height',
    'hcc':   'High cloud cover',
    'lcc':   'Low cloud cover',
    'mcc':   'Medium cloud cover',
    'tcc':   'Total cloud cover',
    'tciw':  'Total column cloud ice water',
    'tclw':  'Total column cloud liquid water',
    'viiwd': 'Vertical integral of divergence of cloud frozen water flux',
    'vilwd': 'Vertical integral of divergence of cloud liquid water flux',
    'viiwe': 'Vertical integral of eastward cloud frozen water flux',
    'vilwe': 'Vertical integral of eastward cloud liquid water flux',
    'viiwn': 'Vertical integral of northward cloud frozen water flux',
    'vilwn': 'Vertical integral of northward cloud liquid water flux',
    'cp':    'Convective precipitation',
    'crr':   'Convective rain rate',
    'ilspf': 'Instantaneous large-scale surface precipitation fraction',
    'lsrr':  'Large scale rain rate',
    'lsp':   'Large-scale precipitation',
    'lspf':  'Large-scale precipitation fraction',
    'mxtpr': 'Maximum total precipitation rate since previous post-processing',
    'mntpr': 'Minimum total precipitation rate since previous post-processing',
    'ptype': 'Precipitation type',
    'tcrw':  'Total column rain water',
    'csf':   'Convective snowfall',
    'csfr':  'Convective snowfall rate water equivalent',
    'lssfr': 'Large scale snowfall rate water equivalent',
    'lsf':   'Large-scale snowfall',
    'asn':   'Snow albedo',
    'rsn':   'Snow density',
    'sd':    'Snow depth',
    'es':    'Snow evaporation',
    'sf':    'Snowfall',
    'smlt':  'Snowmelt',
    'tsn':   'Temperature of snow layer',
    'tcsw':  'Total column snow water'
}

def file_to_grib (home_path: Path):
    
    """
    Rename every file under each mountain subfolder from its current suffix
    to “.grib”, for downstream pygrib processing.
    """

    for mountain_dir in home_path.iterdir():
        if not mountain_dir.is_dir():
            continue
        
        for file in mountain_dir.iterdir():
            if not file.is_file() or file.name.startswith("."):
              continue
              
            new_path = file.with_suffix(".grib")
            print(f"Renaming: {file} -> {new_path}\n")
            file.rename(new_path)

def get_variable_mapping (grib_path: Path):

  """
  Open a single GRIB file and return a mapping from pygrib shortName codes
  to their human-readable variable names. The weather_variable_list variables are not
  the ones pygrib uses internally, so we make the corresponding mapping with the abbreviations
  """

  grbs = pygrib.open(str(grib_path))
  mapping = {msg.shortName: msg.name for msg in grbs}
  grbs.close()
  return mapping

def process_pygrib(grib_path: Path,
                    vars_to_keep: list[str]
                    ) -> pd.DataFrame:
    
  """
  Extract specified variables from a GRIB into a time-indexed DataFrame.
    
  Opens the file with pygrib, filters messages by shortName, computes each
  message’s spatial mean, pivots into columns, and sorts by timestamp.
  """

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
     raise ValueError(f"No records found in {grib_path}")
  
  df = pd.DataFrame(records)
  # Each shortName (variable) is a column
  df = df.pivot_table(index='time', columns='variable', values='value')
  df = df.sort_index()
  return df

def process_grib_to_csv (era5_root: Path,
                         csv_root: Path,
                         vars_to_keep: list[str],
                         ) -> None:
   
  """
  Walk one or more mountain directories of .grib files, process each via
  process_pygrib, and write out parallel .csv files in csv_root.
  
  If era5_root contains .grib files directly, it treats it as a single
  mountain; otherwise it iterates over subdirectories.
  """

  if any(era5_root.glob("*.grib")):
    mountain_dirs = [era5_root]
  else:
    mountain_dirs = [dir for dir in era5_root.iterdir() if dir.is_dir()]

  for mountain_dir in mountain_dirs:

    out_dir = csv_root / mountain_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for grib_file in mountain_dir.glob('*.grib'):
        
        out_csv = out_dir / f"{grib_file.stem}.csv"
        if out_csv.exists():
          print(f"{out_csv.name} already exists. Skipping. ")
          continue

        try:
          df = process_pygrib(grib_file, vars_to_keep)
          df.to_csv(out_csv)
          print(f"Saved: {out_csv}")
        except ValueError as e:
          print(f"Skipping {grib_file}: {e}")

def merge_weather_csvs(mountain_name: str,
                       processed_root: Path = Path('data/era5_data/processed_csvs'),
                       instances_root: Path = Path('data/era5_data/instances/raw_instances')
                        ) -> None:
    """
    Merge all per-5-year CSVs for the given mountain into a single CSV
    and save as instances/{mountain_name}.csv.
    """
    
    mountain_dir = processed_root / mountain_name

    if not mountain_dir.exists() or not mountain_dir.is_dir():
        raise FileNotFoundError(f"No processed CSV directory found for {mountain_name} at {mountain_dir}")

    # Read and concatenate all CSVs
    df_list = []
    for csv_file in sorted(mountain_dir.glob("*.csv")): # Important to sort it just in case since it is a time-series
        df = pd.read_csv(csv_file, parse_dates=['time'])
        df.set_index('time', inplace=True)
        df_list.append(df)

    if not df_list:
        raise ValueError(f"No CSV files found for {mountain_name} at {mountain_dir}")

    merged = pd.concat(df_list)
    merged = merged[~merged.index.duplicated(keep='first')]
    merged.sort_index(inplace=True)

    instances_root.mkdir(parents=True, exist_ok=True)  # Ensure instances directory exists
    mountain_name = mountain_name.replace(" ", "-")
    out_file = instances_root / f"{mountain_name}.csv"

    merged.to_csv(out_file)
    print(f"Merged {len(df_list)} files for '{mountain_name}' into {out_file} ({len(merged)} rows)")

# Second step, obtain a single instance per day. Days are split into two instances at 00:00 and 18:00. Merge into one since the features one has the other lacks
# When merging all the .csv files together, there are two instances per day at 00:00 and 18:00 with complementary features. This function aims at merging these two instances into one, halving the amount of instances and making it more concise
def merge_daily_instances(input_csv: Path,
                          output_csv: Path
                          ) -> None:
  """
   Read a weather CSV with two daily samples (00:00 and 18:00), merge them into a single daily record
   by carrying forward and backward non-null values, and write out a new CSV indexed by date.
  """

  df = pd.read_csv(input_csv, parse_dates=['time'])
  df.set_index('time', inplace=True)
  df.sort_index(inplace=True)

  # We combine the two daily readings by forward and back filling non-null values
  df_daily = df.groupby(df.index.date).apply(lambda g: g.ffill().bfill().iloc[0])
  # We redefine the index to be only the date and no longer the time
  df_daily.index.name = 'date'
  df_daily.index = pd.to_datetime(df_daily.index)

  df_daily.to_csv(output_csv)
  print(f"Merged {input_csv.name} into {output_csv.name}: {len(df)} -> {len(df_daily)} rows")

# Third step, inject peakid as it is the same code mapping as the tabular dataset. Use the peak and sub-peak ids for this

def build_event_instances (tabular_df: pd.DataFrame,
                           instances_root: Path,
                           n_context_days: int = 10
                           ) -> pd.DataFrame:
   """
    Build ML-ready instances for each expedition: for each row in tabular_df,
    map the peakid (or subpeakid) to its parent peak, extract weather data from
    the parent-peak-specific daily CSV for the target date and n_context_days prior,
    flatten into features, and include the target label.

    Args:
        tabular_df: DataFrame with ['PEAKID','SMTDATE','Target'] (SMTDATE datetime).
        instances_root: Path to directory containing {parent_peakid}_daily.csv files.
        n_context_days: Number of days of historical context before event_date.

    Returns:
        DataFrame: one row per expedition event with columns:
          - 'PEAKID' (original code), 'parent_peakid', 'event_date', 'Target'
          - weather features flattened as '<feature>_t-0', '<feature>_t-1', … '<feature>_t-n'
    """
   
   # Maps all 8K peaks and sub-peaks to the same peakid to tie the tabular and weather data since we have the weather data of the main peak for itself and the sub-peaks
   peakid_map = {
      'ANN1': 'ANN1', # Annapurna 1 and its sub-peaks
      'ANNM': 'ANN1',
      'ANNE': 'ANN1',
      'CHOY': 'CHOY', # Cho Oyu
      'DHA1': 'DHA1', # Dhaulagiri I
      'EVER': 'EVER', # Everest
      'KANG': 'KANG', # Kangchenjunga and its sub-peaks
      'KANC': 'KANG',
      'KANS': 'KANG',
      'YALU': 'KANG',
      'YALW': 'KANG',
      'LHOT': 'LHOT', # Lhotse and its sub-peaks
      'LSHR': 'LHOT',
      'LHOM': 'LHOT', 
      'MAKA': 'MAKA', # Makalu 
      'MANA': 'MANA', # Manaslu 
   }

   # Maps the peakids to the peak names to get the right weather files
   peak_names = {
      'ANN1': 'Annapurna-I',
      'CHOY': 'Cho-Oyu',
      'DHA1': 'Dhaulagiri-I',
      'EVER': 'Everest',
      'KANG': 'Kangchenjunga',
      'LHOT': 'Lhotse',
      'MAKA': 'Makalu',
      'MANA': 'Manaslu'
   }
   
   records = []

   for _, row in tabular_df.iterrows():
      raw_peak = row['PEAKID']
      parent_peak = peakid_map.get(raw_peak, raw_peak)

      mountain_name = peak_names.get(parent_peak)

      if mountain_name is None:
         print(f"Warning: No filename mapping for peak code {parent_peak}")
         continue
      

      weather_file = instances_root / f"{mountain_name}.csv"
      print(f"Weather file is {weather_file}\nDoes it exist? {weather_file.exists()}")

      if not weather_file.exists():
         print(f"Warning: Missing weather file {weather_file.name} for {parent_peak}")

      event_date = row['SMTDATE']
      target = row['Target']

      # print(f"PeakId: {raw_peak}\nParent Peak: {parent_peak}\nEvent date: {event_date}\nTarget: {target}\n")

# Fourth step, for each expedition in the tabular table (one row with peakid, date, success) pull the prior 10 daily rows from {mountain}.csv 
# This will be the final ML ready table (one row per expedition with label and weather inputs). Next step will be to do the splits and dataloaders

############### 

era5_path = Path('data/era5_data/database_files')
output_path = Path('data/era5_data/processed_csvs')
output_path.mkdir(exist_ok=True, parents=True)

# Transform all the raw files to .grib for processing
# file_to_grib(era5_path)

# Obtain the variable mapping of the abbreviations pygrib uses internally
# sample = next(era5_path.iterdir()) / (next(era5_path.iterdir()).glob("*.grib").__next__().name)
# mapping = get_variable_mapping(sample)
mapping = weather_mapping 

# Transformation of the .grib files into a more usable .csv format based on the variable mapping
vars_to_keep = list(mapping.keys())
# process_grib_to_csv(era5_path, output_path, vars_to_keep)

instances_path = Path('data/era5_data/instances')
instances_path.mkdir(exist_ok=True, parents=True)
raw_instances_path = Path('data/era5_data/instances/raw_instances')
raw_instances_path.mkdir(exist_ok=True, parents=True)
processed_path = Path('data/era5_data/processed_csvs')

# Merging the multiple weather .csv files into a single one for training feasibility
# merge_weather_csvs("Dhaulagiri I")

merged_instances_path = Path('data/era5_data/instances/merged_instances')
merged_instances_path.mkdir(exist_ok=True, parents=True)
# Merging the two daily readings (00:00 and 18:00) into a single daily reading
# merge_daily_instances(Path('data/era5_data/instances/raw_instances/Dhaulagiri-I.csv'),
#                       Path('data/era5_data/instances/merged_instances/Dhaulagiri-I.csv'))

# Building the instances where we look up expeditions on the tabular dataset, match on the date, and inject the peakid and target variable. Furthemore, we get 10 days of context in the instance as well
tabular_data_path = Path('data/himalayas_data/processed_himalaya_data.csv')
tabular_df = pd.read_csv(tabular_data_path, parse_dates=['SMTDATE'])
build_event_instances(tabular_df, merged_instances_path, 10)
# instances_df = build_event_instances(tabular_df, merged_instances_path, 10)
# instances_output_path = Path('data/era5_data/instances')
# instances_output_path.mkdir(parents=True, exist_ok=True)
# instances_output_file = instances_output_path / 'instances_event_window.csv'
# instances_df.to_csv(instances_output_file, index=False)
# print(f"Wrote {len(instances_df)} event instances to {instances_output_file}")
