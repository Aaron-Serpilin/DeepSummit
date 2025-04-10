import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
import os
import zipfile
from pathlib import Path
import requests
from sklearn.model_selection import train_test_split

def set_seeds(seed: int=42):
    """
    Sets random seeds for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_data_splits (X, y, base_dir: Path, seed:int=42):
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
    