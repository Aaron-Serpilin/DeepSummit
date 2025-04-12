import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class TabularDataset(Dataset):

    def __init__(self,
                csv_file:Path,
                target_column:str,
                cat_cols:list[str], 
                task:str='clf',
                continuos_mean_std=None,
                transform=None):
        
        """
        Class that carry outs the data preparation for the Tabular Dataset.

        Args:
            csv_file: Path to the CSV file
            target_column: Name of the target column
            cat_cols: List of column indices or names of categorical features
            task: Type of problem, either classification or regression
            continuos_mean_std: Tuple of (mean, std) for continuous features
            transform: Transformation to apply to the data
        """
        
        self.data = pd.read_csv(csv_file)
        self.target_column = target_column
        self.transform = transform
        self.features = self.data.drop(columns=[self.target_column])
        self.target = self.data[self.target_column]

        # Mask of ones with the same shape as the features to signal prioritized/ignored features
        self.mask = np.ones(self.features, dtype=np.int64)


    def __len__ (self):
        return len(self.data)
    
    def __getitem__ (self, idx):
        x = self.features.iloc[idx].values.astype('float32')
        y = self.target.iloc[idx]

        if self.transform:
            x = self.transform(x)

        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y
    
himalayan_train_dir = Path("data/himalayas_data/train")
himalayan_val_dir = Path("data/himalayas_data/val")
himalayan_test_dir = Path("data/himalayas_data/test")

himalayan_train_file = himalayan_train_dir / "train.csv"
himalayan_val_file = himalayan_val_dir / "val.csv"
himalayan_test_file = himalayan_test_dir / "test.csv"

df_train = pd.read_csv(himalayan_train_file)

# print(f"First 10 rows:\n{df_train.head(10)}")
# print(f"First training instance:\n{df_train.iloc[0]}\n")
# print(f"Instance shape:\n{df_train.iloc[0].shape}\n")

categorical_columns = ['SEX', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1', 'SEASON', 'O2USED']
continuous_columns = ['CALCAGE', 'HEIGHTM', 'MDEATHS', 'HDEATHS', 'SMTMEMBERS', 'SMTHIRED']

calcage_mean = df_train['CALCAGE']
print(calcage_mean)

# test_set = TabularDataset(himalayan_train_file,
#                           'Target',
#                           ['SEX', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1', 'SEASON', 'O2USED'],
#                           'clf',
#                           )

    