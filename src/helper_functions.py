# Standard Library Imports
import os
import zipfile
from pathlib import Path

# Third-Party Scientific Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from typing import Type, List, Tuple

# PyTorch Imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Torchvision (for image datasets and transforms)
from torchvision import datasets, transforms

# Scikit-Learn for Data Splitting and Other Utilities
from sklearn.model_selection import train_test_split

# Imports from project directories
from tab_transformer.tab_utils import TabularDataset

### Randomization ###

def set_seeds(seed: int=42):
    """
    Sets random seeds for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

### Data Preparation ###

def set_data_splits (X, 
                     y, 
                     base_dir: Path, 
                     seed:int=42):
    """"
    Splits the data into train (80%), validation (10%), and test (10%) sets.
    Afterwards, it saves them as CSV files in the specified directory.

    Args:
        X: Feature matrix
        y: Target vector
        base_dir: Directory where the splits will be saved
        seed: Random seed for reproducibility
    """""
    
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    train_set = pd.concat([X_train, y_train], axis=1)
    val_set   = pd.concat([X_val, y_val], axis=1)
    test_set  = pd.concat([X_test, y_test], axis=1) 

    train_file = train_dir / "train.csv"
    val_file   = val_dir   / "val.csv"
    test_file  = test_dir  / "test.csv"

    train_set.to_csv(train_file, index=False)
    val_set.to_csv(val_file, index=False)
    test_set.to_csv(test_file, index=False)  

    print(f"[INFO] Training set saved to: {train_file}")
    print(f"[INFO] Validation set saved to: {val_file}")
    print(f"[INFO] Test set saved to: {test_file}")

def create_dataloaders (train_file: Path, 
                        val_file: Path, 
                        test_file: Path, 
                        cat_cols: List,
                        continuous_mean_std: List[Tuple[float, float]],
                        target_column: str= 'Target',
                        batch_size: int=32, 
                        num_workers: int=os.cpu_count()):
    
    """
    Creates the training, validation, and testing DataLoaders from CSV files using TabularDataset. 

    It takes in a training, validation, and testing file and turns them 
    into Tabular Datasets and then into PyTorch DataLoaders.

    Args:
        train_file (str): Path to the training CSV file.
        val_file (str): Path to the validation CSV file.
        test_file (str): Path to the testing CSV file.
        cat_cols (list): List of categorical column names.
        continuous_mean_std: List of (mean, std) tuples for continuous features.
        target_column (str): Name of the target column.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes to use for data loading.
    
    Returns:
        A tuple (train_dataloader, val_dataloader, test_dataloader).
    """

    train_data = TabularDataset(csv_file=Path(train_file), target_column=target_column, cat_cols=cat_cols, task='clf', continuous_mean_std=continuous_mean_std)
    val_data = TabularDataset(csv_file=Path(val_file), target_column=target_column, cat_cols=cat_cols, task='clf', continuous_mean_std=continuous_mean_std)
    test_data = TabularDataset(csv_file=Path(test_file), target_column=target_column, cat_cols=cat_cols, task='clf', continuous_mean_std=continuous_mean_std)
    
    train_dataloader = DataLoader(
      train_data, 
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_data, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (train_dataloader, val_dataloader, test_dataloader)