import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

class TabularDataset(Dataset):

    def _init__(self,
                csv_file,
                target_column,
                transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.target_column = target_column
        self.transform = transform
        self.features = self.data.drop(columns=[self.target_column])
        self.target = self.data[self.target_column]

    def __len__ (self):
        return len(self.data)
    
    def __getitem__ (self, idx):
        x = self.features.iloc[idx].values.astype('float32')
        y = self.targets.iloc[idx]

        if self.transform:
            x = self.transform(x)

        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y
    
        
