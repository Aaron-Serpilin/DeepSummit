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
                 transform: transforms.Compose = None
                 ):
        
        """
        A Dataset for past‐7‐day weather → summit‐success classification.

        Args:
            csv_file: Path to your flattened CSV (one row = history + label).
            target_column: Name of the column holding the 0/1 label.
            transform: Desired transformations to be applied on the data.
        """
        
        self.data = pd.read_csv(csv_file, parse_dates=True)
        self.target_column = target_column
        self.transform = transform

        metadata_cols = ["PEAKID", "parent_peakid", "event_date"]
        excluded_columns = metadata_cols + [target_column]
        self.all_columns = self.data.drop(columns=excluded_columns)
        self.num_features = len(self.all_columns)

        pattern = re.compile(r"^(?P<feat>.+)_t-(?P<off>\d+)$")
        features = set()
        offsets = set()

        for column in self.all_columns:
            feature = pattern.match(column)

            if not feature:
                raise ValueError(f"Column {column!r} doesn't match '<feat>_t-<off>' format")
            
            features.add(feature.group("feat"))
            offsets.add(int(feature.group("off")))

        self.features = features
        self.offsets = sorted(offsets)
        self.num_days = len(self.offsets)
        self.num_feats_per_day = len(self.features)

        # print(f"Features are: {features}")
        # print(f"Offsets are: {offsets}")

        # Mask to signal priorities/ignored features
        self.mask = np.ones(self.all_columns.shape, dtype=np.int64)

        # [ cls ] token
        self.cls = np.array([1], dtype=np.int64)

        ### Later add another mask for pressure and most relevant features according to other research papers

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__ (self, idx: int):

        row = self.data.iloc[idx]

        # Flat weather vector rehsaped to [num_days, num_feats_per_day]
        flat_row = row[self.features].values.astype(np.float32)
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
