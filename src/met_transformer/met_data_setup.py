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

# For convenience instead of running get_variable_mapping each time which takes ~3 minutes on CPU
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

def build_event_instances (tabular_df: pd.DataFrame,
                           instances_root: Path,
                           n_context_days: int = 7,
                           max_events: int = None
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
      max_events: If set, process only the first max_events rows from tabular_df (for debugging).

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
  
  # Loading all parents peak's weather DataFrames
  weather_dfs = {}

  for parent_code, name in peak_names.items():
    file_path = instances_root / f"{name}.csv"

    if file_path.exists():
      df = pd.read_csv(file_path, parse_dates=['date'], index_col='date', engine='python')
      weather_dfs[parent_code] = df
    else:
      print(f"[Warning] Missing weather file for {parent_code}: {file_path.name}")
   
  # For debugging purposes to limit the number of instances created
  if max_events is not None:
    iter_df = tabular_df.head(max_events)
  else:
     iter_df = tabular_df

  records = []
 
  # Iterating through each tabular expedition row
  for idx, row in iter_df.iterrows():

    raw_peak = row['PEAKID']
    parent_peak = peakid_map.get(raw_peak, raw_peak)
    mountain_df = weather_dfs.get(parent_peak)
    # print(f"Raw peak is {raw_peak}\nParent peak is {parent_peak}\nMountain df is {mountain_df}")

    if mountain_df is None:
      print(f"[Skipping] No weather DataFrame loaded for peak {parent_peak}")
      continue

    event_date = row['SMTDATE']
    target = row['Target']

    # Context window extraction
    start_date = event_date - pd.Timedelta(days=n_context_days)
    window = mountain_df.loc[start_date:event_date]

    # We use n_context_days + 1 given that the context is the day itself and the week before
    if len(window) != n_context_days + 1:
       print(f"[Skipping] Incomplete window for {raw_peak} on {event_date}: {len(window)} days")
       continue
    
    # print(f"Window for {raw_peak} on {event_date} is {window}")

    record = {
       'PEAKID': raw_peak,
       'parent_peakid': parent_peak,
       'event_date': event_date,
       'Target': target
    }

    for offset, date in enumerate(sorted(window.index, reverse=True)):
      day_series = window.loc[date]

      for feature, value in day_series.items():
        record[f"{feature}_t-{offset}"] = value

    records.append(record)
        
  return pd.DataFrame(records)
       
def load_era5_data (do_file_to_grib: bool = False,
                    do_variable_mapping: bool = False,
                    do_process_grib_to_csv: bool = False,
                    do_merge_weather: bool = False,
                    do_merge_daily: bool = False,
                    do_build_instances: bool = False,
                    mountains: list[str] = None,
                    n_context_days: int = 7
                    ) -> pd.DataFrame | None:

  """

  Carries out the entire ERA5 preparation, and build the corresponding ML-instance .csv file.
  This function however does not request the data from the API directly as that is carried out in the get_data.py file.

  Steps (controlled by flags):
    1. file_to_grib: rename raw files to .grib
    2. variable_mapping: inspect one .grib for shortName->name mapping to get the relevant features and features that pygrib uses internally
    3. process_grib_to_csv: extract vars to .csv files per each 5-year batch
    4. merge_weather: conact the 5-year .csv batches into a full time-series per mountain
    5. merge_daily: collapses the 00:00/18:00 instances into a single daily row
    6. build_instances: join with the tabular dataset .csv, and extract n_context_days+1 window per event and flatten

  Args:
    do_file_to_grib:           run step 1
    do_variable_mapping:       run step 2
    do_process_grib_to_csv:    run step 3
    do_merge_weather:          run step 4
    do_merge_daily:            run step 5
    do_build_instances:        run step 6
    mountains:                 list of mountain CSV names (e.g. ["Everest", …]); defaults to all eight peaks
    n_context_days:            days of history before each event (default 7)

  Returns:
    DataFrame of ML-ready instances (if do_build_instances), else None.

  """

  era5_root = Path('data/era5_data/database_files/raw_files')
  processed_path = Path('data/era5_data/database_files/processed_csvs')
  raw_instances_path = Path('data/era5_data/database_files/instances/raw_instances')
  merged_instances_path = Path('data/era5_data/database_files/instances/merged_instances')

  era5_root.mkdir(exist_ok=True, parents=True)
  processed_path.mkdir(exist_ok=True, parents=True)
  raw_instances_path.mkdir(exist_ok=True, parents=True)
  merged_instances_path.mkdir(exist_ok=True, parents=True)

  if mountains is None:
     mountains = ["Annapurna-I", "Cho-Oyu", "Dhaulagiri-I", "Everest", "Kangchenjunga", "Lhotse", "Makalu", "Manaslu"]
    
  # Step 1
  if do_file_to_grib:
     file_to_grib(era5_root)

  # Step 2
  if do_variable_mapping:
     sample = next(era5_root.iterdir()) / next(next(era5_root.iterdir()).glob("*.grib")).name
     mapping = get_variable_mapping(sample)
  else: 
     mapping = weather_mapping

  # Step 3
  if do_process_grib_to_csv:
     vars_to_keep = list(mapping.keys())
     process_grib_to_csv(era5_root, processed_path, vars_to_keep)

  # Step 4
  if do_merge_weather:
     for mountain in  mountains:
        merge_weather_csvs(mountain,
                           processed_path,
                           raw_instances_path)

  # Step 5
  if do_merge_daily: 
     for mountain in mountains:
        raw_csv = raw_instances_path / f"{mountain}.csv"
        daily_csv = merged_instances_path / f"{mountain}.csv"
        merge_daily_instances(raw_csv, daily_csv)

  # Step 6
  if do_build_instances:
     tabular_data_path = Path('data/himalayas_data/processed_himalaya_data.csv')
     tabular_df = pd.read_csv(tabular_data_path, parse_dates=['SMTDATE'])
     return build_event_instances(tabular_df, merged_instances_path, n_context_days)
  
  return None