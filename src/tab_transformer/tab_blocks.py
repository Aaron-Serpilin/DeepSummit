import torch
from torch import nn, einsum
import torch.nn.functional as F

### Helpers ###

def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None

### Blocks ### 

class MLP (nn.Module):

    def __init__ (self,
                  dims, 
                  act = None
    ):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class simple_MLP(nn.Module):

    def __init__(self,
                 dims):

        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
class sep_MLP(nn.Module):

    def __init__(self,
                 dim,
                 len_feats,
                 categories):
        
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim,5*dim, categories[i]]))

        
    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred
    
class GEGLU (nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

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

