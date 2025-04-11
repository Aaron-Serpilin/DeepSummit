import torch
from torch import nn, einsum
import torch.nn.functional as F

### Helpers ###

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

### Classes ###

# Abstracts all the ski/residual connections into a single class
class Residual (nn.Module):

    def __init__(self, fn):

        super().__init__()
        self.fn = fn

    def forward (self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class PreNorm (nn.Module):

    def __init__(self, dim, fn):

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward (self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

### Transformer ###

# The following transformer class corresponds to equations 1 and 2 that tackle intrasample relationships

# The following transformer class corresponds to equations 3 and 4 that tackle intersample relationships


