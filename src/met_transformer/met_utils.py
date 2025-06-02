import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import numpy as np
from typing import List, Tuple

variables_long = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    # "sea_surface_temperature",
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
]

long_to_short = {
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "2m_dewpoint_temperature": "2d",
    "2m_temperature": "2t",
    "mean_sea_level_pressure": "msl",
    # "sea_surface_temperature": "sst",
    "surface_pressure": "sp",
    "total_precipitation": "tp",
    "ice_temperature_layer_1": "istl1",
    "ice_temperature_layer_2": "istl2",
    "ice_temperature_layer_3": "istl3",
    "ice_temperature_layer_4": "istl4",
    "maximum_2m_temperature_since_previous_post_processing": "mx2t",
    "minimum_2m_temperature_since_previous_post_processing": "mn2t",
    "skin_temperature": "skt",
    "100m_u_component_of_wind": "100u",
    "100m_v_component_of_wind": "100v",
    "10m_u_component_of_neutral_wind": "u10n",
    "10m_v_component_of_neutral_wind": "v10n",
    "10m_wind_gust_since_previous_post_processing": "10fg",
    "instantaneous_10m_wind_gust": "i10fg",
    "cloud_base_height": "cbh",
    "high_cloud_cover": "hcc",
    "low_cloud_cover": "lcc",
    "medium_cloud_cover": "mcc",
    "total_cloud_cover": "tcc",
    "total_column_cloud_ice_water": "tciw",
    "total_column_cloud_liquid_water": "tclw",
    "vertical_integral_of_divergence_of_cloud_frozen_water_flux": "viiwd",
    "vertical_integral_of_divergence_of_cloud_liquid_water_flux": "vilwd",
    "vertical_integral_of_eastward_cloud_frozen_water_flux": "viiwe",
    "vertical_integral_of_eastward_cloud_liquid_water_flux": "vilwe",
    "vertical_integral_of_northward_cloud_frozen_water_flux": "viiwn",
    "vertical_integral_of_northward_cloud_liquid_water_flux": "vilwn",
    "convective_precipitation": "cp",
    "convective_rain_rate": "crr",
    "instantaneous_large_scale_surface_precipitation_fraction": "ilspf",
    "large_scale_rain_rate": "lsrr",
    "large_scale_precipitation": "lsp",
    "large_scale_precipitation_fraction": "lspf",
    "maximum_total_precipitation_rate_since_previous_post_processing": "mxtpr",
    "minimum_total_precipitation_rate_since_previous_post_processing": "mntpr",
    "precipitation_type": "ptype",
    "total_column_rain_water": "tcrw",
    "convective_snowfall": "csf",
    "convective_snowfall_rate_water_equivalent": "csfr",
    "large_scale_snowfall_rate_water_equivalent": "lssfr",
    "large_scale_snowfall": "lsf",
    "snow_albedo": "asn",
    "snow_density": "rsn",
    "snow_depth": "sd",
    "snow_evaporation": "es",
    "snowfall": "sf",
    "snowmelt": "smlt",
    "temperature_of_snow_layer": "tsn",
    "total_column_snow_water": "tcsw"
}

def check_csv_features(csv_file: str) -> None:
    """
    Loads `csv_file`, extracts the short‐code prefixes before '_t-<offset>' from its column names,
    and prints any variables in `variables_long` whose mapped short code is missing in the CSV.
    
    Args:
        csv_file: Path to the flattened weather CSV.
        variables_long: List of descriptive variable names (length 56).
        long_to_short: Dict mapping each long name in variables_long to its short code prefix.
    """
    # 1) Load CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # 2) Collect all "<short>_t-<offset>" prefixes actually present
    pattern = re.compile(r"^(?P<feat>.+)_t-(?P<off>\d+)$")
    extracted_features = set()
    for col in df.columns:
        m = pattern.match(col)
        if m:
            extracted_features.add(m.group("feat"))
    
    # 3) Print all found short codes, sorted
    print("Features found in CSV (prefix before '_t-<offset>'):")
    for feat in sorted(extracted_features):
        print(f"  {feat}")
    print(f"\nTotal distinct prefixes extracted: {len(extracted_features)}\n")
    
    # 4) Compare against the provided long‐name list
    missing_shorts = []
    for long_name in variables_long:
        short = long_to_short.get(long_name)
        if short is None:
            # No mapping provided for this long_name
            missing_shorts.append((long_name, None))
        elif short not in extracted_features:
            missing_shorts.append((long_name, short))
    
    if missing_shorts:
        print("The following long‐names map to short codes not found in the CSV:")
        for long_name, short in missing_shorts:
            if short is None:
                print(f"  {long_name!r} → no short‐code mapping provided")
            else:
                print(f"  {long_name!r}  →  expected prefix '{short}'  (MISSING)")
    else:
        print("All expected short codes appear in the CSV.")

class WeatherDataset (Dataset):

    def __init__(self,
                 csv_file: Path,
                 target_column: str,
                 priority_features: List[str] = None,
                 metadata_cols: List[str] = None,
                 continuous_mean_std: List[Tuple[float, float]] = None,
                 transform: transforms.Compose = None,
                 variables: List[str] = None
                 ):
        
        """
        A Dataset for past‐7‐day weather → summit‐success classification.

        Args:
            csv_file: Path to your flattened CSV (one row = history + label).
            target_column: Name of the column holding the 0/1 label.
            continuous_mean_std: List of (mean, std) tuples for each continuous feature. If provided, continuous features are normalized
            transform: Desired transformations to be applied on the data.
        """
        
        self.data = pd.read_csv(csv_file, parse_dates=True)

        # check_csv_features(csv_file=csv_file)

        # Params
        self.target_column = target_column
        self.transform = transform

        if metadata_cols is None:
            metadata_cols = ["PEAKID", "parent_peakid", "event_date"]

        if variables is None:
            raise ValueError("You must pass a variable list")

        excluded_columns = set(metadata_cols + [target_column])

        # Features DataFrame and column list
        self.feature_df = self.data.drop(columns=excluded_columns, errors="ignore") # DataFrame containing all of the values for every flattened feature column
        self.num_features = len(self.feature_df)
        self.num_flat_features = len(self.feature_df.columns)
        self.feature_cols = list(self.feature_df.columns) # flattened column names with the the feature and time offset

        # Parsing base feature names and day offsets
        pattern = re.compile(r"^(?P<feat>.+)_t-(?P<off>\d+)$")
        features = set()
        offsets = set()

        # print(f"Features: {len(self.feature_df)}")
      
        for column in self.feature_df:

            feature = pattern.match(column)
            if not feature:
                raise ValueError(f"Column {column!r} doesn't match '<feat>_t-<off>' format")
            
            features.add(feature.group("feat"))
            offsets.add(int(feature.group("off")))

        # print(f"Features: {features}\nOffsets: {offsets}\n")

        # Metadata
        self.base_features = sorted(features) # flattened column names without time offsets
        self.test = list(variables)
        # print(f"Test base: {self.test}\nOriginal base: {self.base_features}\nTest vs Original length: {len(self.test)} vs {len(self.base_features)}\n")
        # print(f"Base features: {len(self.base_features)}")
        self.offsets = sorted(offsets)
        self.num_days = len(self.offsets)
        self.num_feats_per_day = len(self.base_features)

        # Normalization
        if continuous_mean_std is None:
            stats = self.feature_df.agg(["mean", "std"])
            means = stats.loc["mean"].values.astype(np.float32)
            stds = stats.loc["std"].values.astype(np.float32)
            continuous_mean_std = list(zip(means, stds))

        self.means = np.array([mean for mean, std in continuous_mean_std], dtype=np.float32)
        self.stds = np.array([std for mean, std in continuous_mean_std], dtype=np.float32)
        self.stds[self.stds == 0] = 1.0 # we do this to avoid division by 0 problems

        # Mask to signal priority features
        self.priority_features = priority_features
        self.mask = np.array([1 if feat in priority_features else 0 for feat in self.base_features], dtype=np.int64)

        # Intra-sample masks for multiple smaller views of the context days
        seq_len = 1 + self.num_days # we add 1 due to the [ cls ] token

        self.window_masks = np.stack([
            np.concatenate([
                # self.cls,
                np.ones(intra_context, dtype=np.int64),
                np.zeros(self.num_days - intra_context, dtype=np.int64)
            ])

            for intra_context in range(1, seq_len)
        ], axis=0) 

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__ (self, idx: int):

        row = self.data.iloc[idx]

        # Flat weather vector reshaped to [num_days, num_feats_per_day] 
        flat = row[self.feature_cols].values.astype(np.float32)

        # Normalization
        flat = (flat - self.means) / self.stds

        # Reshaping into (days, feats)
        X_days = flat.reshape(self.num_days, self.num_feats_per_day)

        X = X_days
        mask = self.mask

        # Label
        y = np.int64(row[self.target_column])

        if self.transform:
            X, mask, y = self.transform(X, mask, y)

        # Intra sample mask
        window_masks = self.window_masks

        # Convert arrays to torch tensors
        X_tensor = torch.tensor(X)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        target_tensor = torch.tensor(y)
        window_mask_tensor = torch.tensor(window_masks)

        return X_tensor, mask_tensor, target_tensor, window_mask_tensor

