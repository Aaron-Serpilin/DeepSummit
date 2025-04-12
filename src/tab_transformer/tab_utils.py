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
                task:str='clf',
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
            continuous_mean_std: List of (mean, std) tuples for each continuous feature. If provided, continuous features are normalized.
        """
        
        self.data = pd.read_csv(csv_file)
        self.target_column = target_column

        # Conversion of categorical columns to integer codes
        for col in cat_cols:
            self.data[col] = self.data[col].astype('category').cat.codes

        self.features = self.data.drop(columns=[self.target_column])
        self.target = self.data[self.target_column]

        # Mask of ones with the same shape as the features to signal prioritized/ignored features
        self.mask = np.ones(self.features.shape, dtype=np.int64)

        # While you can access columns by name in DataFrames, NumPy only has integer indices
        cat_indices = [self.data.columns.get_loc(col) for col in cat_cols if col != target_column]
        X_np = self.features.values
        all_indices = set(range(X_np.shape[1]))
        con_indices = sorted(list(all_indices - set(cat_indices)))

        # Categorical (X1) and Continuous (X2) extraction
        self.X1 = X_np[:, cat_indices].copy().astype(np.int64)
        self.X2 = X_np[:, con_indices].copy().astype(np.float32)
        # Masks
        self.X1_mask = self.mask[:, cat_indices].copy().astype(np.int64)
        self.X2_mask = self.mask[:, con_indices].copy().astype(np.int64)
   
        if task == 'clf':
            self.y = self.target.values
        else: 
            self.y = self.tarrget.values.astype(np.float32)

        # [ cls ] token 
        self.cls = np.zeros((len(self.y), 1), dtype=int)
        self.cls_mask = np.ones((len(self.y), 1), dtype=int)

        # Normalization
        if continuous_mean_std is not None:
            means = np.array([mean for mean, std in continuous_mean_std], dtype=np.float32)
            stds = np.array([std for mean, std in continuous_mean_std], dtype=np.float32)
            self.X2 = (self.X2 - means) / stds

    def __len__ (self):
        return len(self.data)
    
    def __getitem__ (self, idx):

        """
        Returns:
            A tuple containing:
                1) The concatenated categorical tensor with the CLS token as a tensor of shape (n_cat+1,)
                2) The continuous features as a tensor.
                3) The target value as a tensor.
                4) The concatenated categorical mask (with CLS token mask) as a tensor.
                5) The continuous mask as a tensor.
        """
    
### Everything underneath is for testing, will be cleaned
    
himalayan_train_file = Path("data/himalayas_data/train/train.csv")
df_file = pd.read_csv(himalayan_train_file)

categorical_columns = ['SEX', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1', 'SEASON', 'O2USED']
continuous_columns = ['CALCAGE', 'HEIGHTM', 'MDEATHS', 'HDEATHS', 'SMTMEMBERS', 'SMTHIRED']
continuous_means = [df_file[col].mean() for col in continuous_columns]
continuous_stds = [df_file[col].std() for col in continuous_columns]
continuous_mean_std = list(zip(continuous_means, continuous_stds))

experiment_set = TabularDataset(himalayan_train_file,
                          'Target',
                          categorical_columns,
                          'clf',
                          continuous_mean_std)

    