import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

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

    for batch, data in enumerate(dataloader):

        # Need to define y
        optimizer.zero_grad()
        x_categ, x_cont, y_true, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

        # Converting data into embeddings
        _, x_categ_emb, x_cont_emb = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model, False) 
        sequence_embeddings = model.transformer(x_categ_emb, x_cont_emb) 

        # Extracting the cls token from each instance
        cls_embeddings = sequence_embeddings[:, 0, :] 

        # Forward pass
        y_pred = model.mlpfory(cls_embeddings) 
        loss = loss_fn(y_pred, y_true)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y_true).sum().item()/len(y_pred)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: torch.nn.Module, 
    device: torch.device 
    ) -> Tuple[float, float]:

    """ 
    Test a PyTorch model for a single epoch

    Turns a target PyTorch model to "eval" mode and then performs a forward pass on a testing dataset

    It returns a tuple of testing loss and testing accuracy metrics
    """

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        for batch, data in enumerate(dataloader):

            x_categ, x_cont, y_true, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

            # Converting data into embeddings
            _, x_categ_emb, x_cont_emb = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model, False) 
            sequence_embeddings = model.transformer(x_categ_emb, x_cont_emb) 

            # Extracting the cls token from each instance
            cls_embeddings = sequence_embeddings[:, 0, :] 

            # Forward pass
            y_pred = model.mlpfory(cls_embeddings) 

            # Calculate and accumulate loss
            loss = loss_fn(y_pred, y_true)
            test_loss = loss.item()

            test_pred_labels = y_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y_true).sum().item()/len(test_pred_labels))
            
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc
    
def train(
    model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader, 
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader, 
    optimizer:torch.optim.Optimizer, 
    loss_fn: torch.nn.Module, 
    epochs: int,
    writer: torch.utils.tensorboard.writer.SummaryWriter
    ) -> Dict[str, List]:

    """ 
    Trains, validates and tests a PyTorch model

    Passes a target PyTorch model through the train_step(), val_step(), and test_step() functions for a number of epochs,
    training, validating and testing the model in the same epoch loop

    It calculates, prints and stores evaluation metrics throughout

    It returns a dictionary of training, validation and testing loss as well as training, validation and testing accuracy metrics.
    Each metric has a value in a list for each epoch
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

        val_loss, val_acc = test_step(
            model=model,
            dataloader=val_dataloader, 
            loss_fn=loss_fn,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
          writer.add_scalars(main_tag="Loss",
                            tag_scalar_dict={"train_loss": train_loss,
                                              "test_loss": test_loss},
                                              global_step=epoch)
          writer.add_scalars(main_tag="Accuracy",
                            tag_scalar_dict={"train_acc": train_acc,
                                              "test_acc": test_acc},
                                              global_step=epoch)

          writer.close()
        else:
          pass

    return results