import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from typing import List, Tuple

class TabularDataset(Dataset):

    def __init__(self,
                csv_file:Path,
                target_column:str,
                cat_cols:list[str], 
                continuous_mean_std:List[Tuple[float, float]]=None):
        
        """
        Class to prepare a tabular dataset for transformer models.
        
        This class loads data from a CSV file, splits it into categorical and continuous parts,
        creates corresponding masks, and normalizes continuous features if parameters are provided.

        Args:
            csv_file: Path to the CSV file
            target_column: Name of the target column
            cat_cols: List of column indices or names of categorical features
            task: Type of problem, either classification or regression
            continuous_mean_std: List of (mean, std) tuples for each continuous feature. If provided, continuous features are normalized
        """
        
        self.data = pd.read_csv(csv_file)
        self.target_column = target_column

        # Conversion of categorical columns to integer codes
        for col in cat_cols:
            self.data[col] = self.data[col].astype('category').cat.codes

        self.features_df = self.data.drop(columns=[self.target_column])
        self.y = self.target.values
        self.target = self.data[self.target_column]
    
        # Mask of ones with the same shape as the features to signal prioritized/ignored features
        self.mask = np.ones(self.features.shape, dtype=np.int64)

        # While you can access columns by name in DataFrames, NumPy only has integer indices
        cat_indices = [self.data.columns.get_loc(col) for col in cat_cols if col != target_column]
        X_np = self.features_df.values
        all_columns = list(self.features_df.columns)
        all_indices = set(range(X_np.shape[1]))
        cont_indices = sorted(list(all_indices - set(cat_indices)))

        # Categorical (X1) and Continuous (X2) extraction
        self.X1 = X_np[:, cat_indices].copy().astype(np.int64)
        self.X2 = X_np[:, con_indices].copy().astype(np.float32)

        # Masks
        self.X1_mask = self.mask[:, cat_indices].copy().astype(np.int64)
        self.X2_mask = self.mask[:, cont_indices].copy().astype(np.int64)

        # Normalization
        if continuous_mean_std is None:
            cont_columns = [all_columns[i] for i in cont_indices]
            stats = self.features_df[cont_columns].agg(["mean", "std"])
            means = stats.loc["mean"].values.astype(np.float32)
            stds = stats.loc["std"].values.astype(np.float32)
            continuous_mean_std = list(zip(means, stds))

        self.means = np.array([mean for mean, std in continuous_mean_std], dtype=np.float32)
        self.stds = np.array([std for mean, std in continuous_mean_std], dtype=np.float32)
        self.stds[self.stds == 0] = 1.0 # we do this to avoid division by 0 problems

        self.X2 = (self.XS - self.means) / self.stds

        # [ cls ] token 
        self.cls = np.zeros((len(self.y), 1), dtype=int)
        self.cls_mask = np.ones((len(self.y), 1), dtype=int)

    def __len__ (self):
        return len(self.y)
    
    def __getitem__ (self, idx: int):

        """
        Returns:
            A tuple containing:
                1) The concatenated categorical tensor with the CLS token as a tensor of shape (n_cat+1,)
                2) The continuous features as a tensor
                3) The target value as a tensor
                4) The concatenated categorical mask (with CLS token mask) as a tensor
                5) The continuous mask as a tensor
        """

        cat_instance = self.X1[idx]         # shape: (n_cat,)
        con_instance = self.X2[idx]           # shape: (n_cont,)
        target_instance = self.y[idx]
        
        cat_mask_instance = self.X1_mask[idx]
        con_mask_instance = self.X2_mask[idx]
        
        # Concatenate CLS token to categorical features and its mask (only append to categorical and not continuous since its appended once per instance)
        cat_instance_with_cls = np.concatenate((self.cls[idx], cat_instance))  # shape: (n_cat + 1,)
        cat_mask_with_cls = np.concatenate((self.cls_mask[idx], cat_mask_instance))  # shape: (n_cat + 1,)
        
        # Convert arrays to torch tensors
        cat_tensor = torch.tensor(cat_instance_with_cls, dtype=torch.long)
        con_tensor = torch.tensor(con_instance, dtype=torch.float)
        target_tensor = torch.tensor(target_instance)
        cat_mask_tensor = torch.tensor(cat_mask_with_cls, dtype=torch.long)
        con_mask_tensor = torch.tensor(con_mask_instance, dtype=torch.long)
        
        return cat_tensor, con_tensor, target_tensor, cat_mask_tensor, con_mask_tensor
    