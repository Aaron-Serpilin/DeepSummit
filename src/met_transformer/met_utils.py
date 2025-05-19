import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import numpy as np
from typing import List, Tuple

class WeatherDataset (Dataset):

    def __init__(self,
                 csv_file: Path,
                 target_column: str,
                 continuous_mean_std: List[Tuple[float, float]] = None,
                 transform: transforms.Compose = None
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
        self.target_column = target_column
        self.transform = transform

        metadata_cols = ["PEAKID", "parent_peakid", "event_date"]
        excluded_columns = metadata_cols + [target_column]
        self.feature_df = self.data.drop(columns=excluded_columns) # DataFrame containing all of the values for every flattened feature column
        self.num_features = len(self.feature_df)
        self.feature_cols = [col for col in self.feature_df if col not in excluded_columns] # flattened column names with the the feature and time offset

        pattern = re.compile(r"^(?P<feat>.+)_t-(?P<off>\d+)$")
        features = set()
        offsets = set()

        for column in self.feature_df:
            feature = pattern.match(column)

            if not feature:
                raise ValueError(f"Column {column!r} doesn't match '<feat>_t-<off>' format")
            
            features.add(feature.group("feat"))
            offsets.add(int(feature.group("off")))

        self.features = features # flattened column names without time offsets
        self.offsets = sorted(offsets)
        self.num_days = len(self.offsets)
        self.num_feats_per_day = len(self.features)

        # Mask to signal priorities/ignored features
        self.mask = np.ones(self.feature_df.shape, dtype=np.int64)

        # [ cls ] token
        self.cls = np.array([1], dtype=np.int64)

        # Stats calculation for normalization
        stats = self.data[self.feature_cols].agg(["mean", "std"])
        self.means = stats.loc["mean"].values.astype(np.float32)
        self.stds = stats.loc["std"].values.astype(np.float32)

        # print(f"means: {len(self.means)}\n")
        # print(f"stds: {len(self.stds)}\n")

        ### Later add another mask for pressure and most relevant features according to other research papers

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

        # Creation [ cls ] token, and prepending it
        cls_token = np.zeros((1, self.num_feats_per_day), dtype=np.float32)
        X = np.vstack([cls_token, X_days]) # shape: (num_days+1, num_feats_per_day)

        # Building full mask [ cls ] + each day
        mask = np.concatenate([self.cls, self.mask], axis=0)

        # Label
        y = np.int64(row[self.target_column])

        if self.transform:
            X, mask, y = self.transform(X, mask, y)

        X_tensor = torch.tensor(X)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        target_tensor = torch.tensor(y)

        return X_tensor, mask_tensor, target_tensor

test_path = Path("data/era5_data/era5_data.csv")
test = WeatherDataset(test_path, 'Target', None)
