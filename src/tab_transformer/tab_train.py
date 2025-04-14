import torch
from torch import nn

from augmentations import permute_data
from tab_model import SAINT
from tab_utils import TabularDataset
from helper_functions import set_seeds

from torch.utils.data import DataLoader

import os
import numpy as np

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seeds(42)

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    device: torch.device
    ) -> Tuple[float, float]:

    """ 
    Trains a PyTorch model for a single epoch

    Turns a target PyTorch model to training mode and then runs through all of the training steps:
        Forward Pass
        Loss Calculation
        Optimizer Step

    It returns a tuple of training loss and training accuracy metrics
    """

    model.train()

    train_loss, train_acc = 0, 0

    return train_loss, train_acc

def test_step(
    model: torch.nn.Module, # model to be tested
    dataloader: torch.utils.data.DataLoader, # DataLoader instance for the model to be tested on
    loss_fn: torch.nn.Module, # loss function to calculate loss on the test data
    device: torch.device 
    ) -> Tuple[float, float]:

    """ 
    Test a PyTorch model for a single epoch

    Turns a target PyTorch model to "eval" mode and then performs a forward pass on a testing dataset

    It returns a tuple of testing loss and testing accuracy metrics
    """

    model.eval()

    test_loss, test_acc = 0, 0

    return test_loss, test_acc
    
def train(
    model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader, 
    test_dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: torch.nn.Module, 
    epochs: int,
    device: torch.device
    ) -> Dict[str, List]:

    """ 
    Trains and tests a PyTorch model

    Passes a target PyTorch model through the train_step() and test_step() functions for a number of epochs,
    training and testing the model in the same epoch loop

    It calculates, prints and stores evaluation metrics throughout

    It returns a dictionary of training and testing loss as well as training and testing accuracy metrics.
    Each metric has a value in a list for each epoch
    """

    results = {
        "train_loss": [],
        "train_acc": [], 
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        
        train_loss, train_acc = train_step(
            model=model, 
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results