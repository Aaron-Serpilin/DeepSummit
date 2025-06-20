import torch
from torch.utils.data import DataLoader

from src.tab_transformer.tab_augmentations import embed_data_mask

def extract_logits_tab(model: nn.Module,
                       dataloader: DataLoader,
                       device: torch.device):
    
    """
    Run the SAINT model over every batch in dataloader,
    returning (logits, labels) all on CPU.
    """

    model.eval()
    all_logits = []
    all_labels = []

    with torch.inference_mode():

        for batch, data in enumerate(dataloader):
            
            x_categ, x_cont, y_true, cat_mask, con_mask = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
                data[3].to(device),
                data[4].to(device),
            )

            _, x_categ_emb, x_cont_emb = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, False)
            sequence_embeddings = model.transformer(x_categ_emb, x_cont_emb)
            cls_embeddings = sequence_embeddings[:, 0, :]
            logits  = model.mlpfory(cls_embeddings) 

            all_logits.append(logits.cpu())
            all_labels.append(y_true.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

def extract_logits_met(model: nn.Module,
                       dataloader: DataLoader,
                       device: torch.device):
    
    """
    Run the Stormer model over every batch in dataloader,
    returning (logits, labels) all on CPU.
    """

    model.eval()
    all_logits = []
    all_labels = []

    with torch.inference_mode():

        for batch, data in enumerate(dataloader):
           
            X, mask, y_true, window_mask = data
            X, mask, y_true, window_mask = X.to(device), mask.to(device), y_true.to(device), window_mask.to(device)

            if torch.isnan(full_seq_pred).any():
                continue

            full_seq_pred = model(X)           
           
            logits = full_seq_pred[:, 0, :]     

            all_logits.append(logits.cpu())
            all_labels.append(y_true.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)
