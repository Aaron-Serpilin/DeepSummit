# Standard Library Imports
import os
from pathlib import Path

# Third-Party Scientific Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from typing import Type, Tuple, Dict, Any

# PyTorch Imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Scikit-Learn for Data Splitting and Other Utilities
from sklearn.model_selection import train_test_split

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

    train_file = train_dir / "train.csv"
    val_file   = val_dir   / "val.csv"
    test_file  = test_dir  / "test.csv"

    if train_file.exists() and val_file.exists() and test_file.exists():
        print(f"[INFO] Data splits already exist under {base_dir}. Skipping split.")
        return
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    train_set = pd.concat([X_train, y_train], axis=1)
    val_set   = pd.concat([X_val, y_val], axis=1)
    test_set  = pd.concat([X_test, y_test], axis=1) 

    train_set.to_csv(train_file, index=False)
    val_set.to_csv(val_file, index=False)
    test_set.to_csv(test_file, index=False)  

    print(f"[INFO] Training set saved to: {train_file}")
    print(f"[INFO] Validation set saved to: {val_file}")
    print(f"[INFO] Test set saved to: {test_file}")

def create_dataloaders (dataset_class: Type,
                        train_file: Path, 
                        val_file: Path, 
                        test_file: Path, 
                        num_workers: int = os.cpu_count(),
                        batch_size: int = 32, 
                        dataset_kwargs: Dict[str, Any] = None # any extra args for the Dataset
                        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    """
    Creates the training, validation, and testing DataLoaders from CSV files using TabularDataset. 

    It takes in a training, validation, and testing file and turns them 
    into Tabular Datasets and then into PyTorch DataLoaders.

    Args:
        dataset_class (Type): Class of Dataset to instantiate
        train_file (str): Path to the training CSV file.
        val_file (str): Path to the validation CSV file.
        test_file (str): Path to the testing CSV file.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes to use for data loading.
        kwargs: Dict[str, Any]: Dictionary of keyword args to pass to the Dataset constructor. These include the following:
            continuous_mean_std: List of (mean, std) tuples for continuous features.
            target_column (str): Name of the target column.
            cat_cols (list): List of categorical column names.
    
    Returns:
        A tuple (train_dataloader, val_dataloader, test_dataloader).
    """

    dataset_kwargs = dataset_kwargs or {}

    train_dataset = dataset_class(csv_file=Path(train_file), **dataset_kwargs)
    val_dataset = dataset_class(csv_file=Path(val_file), **dataset_kwargs)
    test_dataset = dataset_class(csv_file=Path(test_file), **dataset_kwargs)
    
    # When using custom Dataset classes such as Tabular/WeatherDataset, it is important to use DataLoaders
    # when training/evaluating since every time we fetch a sample, it triggers __getitem__ and the subsequent normalization
    train_dataloader = DataLoader(
      train_dataset, 
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (train_dataloader, val_dataloader, test_dataloader)

def plot_loss_curves(results):
    """
    Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]
    test_loss = results["test_loss"]

    train_accuracy = results["train_acc"]
    val_accuracy = results["val_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def save_model(model: torch.nn.Module, # model to save
               target_dir: str, # directory for saving the model to
               model_name: str # filename for the saved model. Should include either ".pth" or ".pt" as the file extension
    ):

    """ 
    Saves a PyTorch model to a target directory
    """

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def create_writer(experiment_name: str,
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():

    """
    Create and return a TensorBoard SummaryWriter that logs under:
        runs/{experiment_name}/{extra}/{timestamp}/

    Args:
        experiment_name (str): top-level folder under runs, e.g. "saint_runs"
        extra (str, optional): details of this run, e.g. "50_epochs_lr_1e-3_depth_4"

    Returns:
        SummaryWriter pointing at runs/... path.
    """

    from datetime import datetime
    import os

    base = "runs"
    timestamp = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")

    if extra:
        log_dir = os.path.join(base, experiment_name, extra, timestamp) 
    else:
        log_dir = os.path.join(base, experiment_name, timestamp) 

    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)
