import torch
import numpy as np
from torch import Tensor

def permute_data (x1:Tensor,
                    x2:Tensor,
                    y=None,
                    alpha=0.2,
                    use_cuda=True
):
    
    """
    Returns mixed inputs with per-sample mixing coefficients.
    
    Args:
        x1: Tensor of (categorical) data embeddings
        x2: Tensor of (continuous) data embeddings
        y: Optional target tensor
        alpha: Parameter for the Beta distribution to sample mixing ratios
        use_cuda: If True, moves variables to the GPU
    
    Returns:
        A tuple (mixed_x1, mixed_x2, y_a, y_b, lam) if y is provided,
        otherwise (mixed_x1, mixed_x2, lam).
        'lam' is a tensor of shape [batch_size] containing per-sample mixing coefficients
    """

    batch_size = x1.size(0)

    # Sampling a lambda for each sample in the bacth from Beta(alpha, alpha)
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample((batch_size,))
    else:
        lam = torch.ones(batch_size)

    # Send lambda to GPU if available. PyTorch tensors have an inherit .device attribute
    lam = lam.to(x1.device) if use_cuda else lam

    # Random permutation of batch indices
    index = torch.randperm(batch_size).to(x1.device)

    # Reshaping lambda by broadcasting 1s across x'1 dimensions
    lam_x1 = lam.view(batch_size, *([1] * (x1.dim() - 1)))
    lam_x2 = lam.view(batch_size, *([1] * (x2.dim() - 1)))
    
    # Mix each sample with index[i]
    mixed_x1 = lam_x1 * x1 + (1 - lam_x1) * x1[index]
    mixed_x2 = lam_x2 * x2 + (1 - lam_x2) * x2[index]

    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b, lam
    
    return mixed_x1, mixed_x2, lam

def embed_data_mask(x_categ:Tensor, 
                    x_cont:Tensor, 
                    cat_mask:Tensor, 
                    con_mask:Tensor, 
                    model:torch.nn.Module, 
                    vision_dset:bool=False):

    """
    Embed raw categorical & continuous inputs and apply mask embeddings.

    Args:
        x_categ: Integer category indices, shape (B, n_cat)
        x_cont: Raw continuous feature values, shape (B, n_cont)
        cat_mask: Binary mask for categorical tokens (1 = keep, 0 = mask), shape (B, n_cat)
        con_mask): Binary mask for continuous tokens (1 = keep, 0 = mask), shape (B, n_cont)
        model: A SAINT model instance, providing the offsets, embedding tables, and mlps
        vision_dset: If True, add positional encodings to x_categ_enc. Default: False

    Returns:
        tuple:
            x_categ: Offset category indices after adding model.categories_offset, shape (B, n_cat)
            x_categ_enc: Embedded (and masked) categorical tokens, shape (B, n_cat, dim)
            x_cont_enc: Embedded (and masked) continuous tokens, shape (B, n_cont, dim)
    """

    device = x_cont.device
    
    # The size is (Batch Size, n_cat + 1) where the +1 is due to the prepended cls token in TabularDataset
    # We only deal with adjusting the offset size of the categorical data since we only appended the cls token once per instance in TabularDataset
    offsets = model.categories_offset
    if offsets.size(0) == x_categ.size(1) - 1:
        zeros = torch.zeros(1, dtype=offsets.dtype, device=x_categ.device)
        offsets = torch.cat([zeros, offsets], dim=0)

    x_categ += offsets.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape

    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')    

    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)
    x_cont_enc = x_cont_enc.to(device)
    
    # Adjustment of the cat offset where we prepend as above to have matching shapes with the cls token concatenation
    cat_off = model.cat_mask_offset
    if cat_off.size(0) == cat_mask.size(1) - 1:
        zeros = torch.zeros(1, dtype=cat_off.dtype, device=cat_mask.device)
        cat_off = torch.cat([zeros, cat_off], dim=0)
    
    cat_mask_temp = cat_mask + cat_off.type_as(cat_mask)

    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        
        pos = np.tile(np.arange(x_categ.shape[-1]),(x_categ.shape[0],1))
        pos =  torch.from_numpy(pos).to(device)
        pos_enc =model.pos_encodings(pos)
        x_categ_enc+=pos_enc

    return x_categ, x_categ_enc, x_cont_enc