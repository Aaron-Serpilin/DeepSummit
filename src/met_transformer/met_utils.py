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
                 metadata_cols: List[str] = None,
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

        # Params
        self.target_column = target_column
        self.transform = transform

        if metadata_cols is None:
            metadata_cols = ["PEAKID", "parent_peakid", "event_date"]

        excluded_columns = set(metadata_cols + [target_column])

        # Features DataFrame and column list
        self.feature_df = self.data.drop(columns=excluded_columns, errors="ignore") # DataFrame containing all of the values for every flattened feature column
        self.num_features = len(self.feature_df)
        self.feature_cols = list(self.feature_df.columns) # flattened column names with the the feature and time offset

        # Parsing base feature names and day offsets
        pattern = re.compile(r"^(?P<feat>.+)_t-(?P<off>\d+)$")
        features = set()
        offsets = set()
      
        for column in self.feature_df:

            feature = pattern.match(column)
            if not feature:
                raise ValueError(f"Column {column!r} doesn't match '<feat>_t-<off>' format")
            
            features.add(feature.group("feat"))
            offsets.add(int(feature.group("off")))

        # Metadata
        self.base_features = sorted(features) # flattened column names without time offsets
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

        # Mask to signal priorities/ignored features
        self.mask = np.ones(self.num_days, dtype=np.int64)

        # [ cls ] token
        self.cls = np.array([1], dtype=np.int64)

        # Intra-sample masks for multiple smaller views of the context days
        seq_len = 1 + self.num_days # we add 1 due to the [ cls ] token
        self.full_mask = np.concatenate([self.cls, self.mask])

        self.window_masks = np.stack([
            np.concatenate([
                self.cls,
                np.ones(intra_context, dtype=np.int64),
                np.zeros(self.num_days - intra_context, dtype=np.int64)
            ])

            for intra_context in range(1, seq_len)
        ], axis=0) 

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

        # Intra sample mask
        window_masks = self.window_masks

        # Convert arrays to torch tensors
        X_tensor = torch.tensor(X)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        target_tensor = torch.tensor(y)
        window_mask_tensor = torch.tensor(window_masks)

        return X_tensor, mask_tensor, target_tensor, window_mask_tensor