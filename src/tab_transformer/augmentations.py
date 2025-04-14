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
