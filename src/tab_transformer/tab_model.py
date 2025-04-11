import torch
from torch import nn, einsum
import torch.nn.functional as F
from tab_attention import FeedForward, Attention
from einops import rearrange

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

### Transformers ###

# The following transformer class corresponds to equations 1 and 2 that tackle intrasample relationships

class Transformer(nn.Module):
    def __init__(self, 
                 num_tokens, 
                 dim, 
                 depth, 
                 heads, 
                 dim_head, 
                 attn_dropout, 
                 ff_dropout):
        
        super().__init__()
        self.layers = nn.ModuleList([])


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
    
# The following transformer class corresponds to equations 3 and 4 that tackle intersample relationships

class RowColTransformer(nn.Module):
    def __init__(self, 
                 num_tokens, 
                 dim, nfeats, 
                 depth, heads, 
                 dim_head, 
                 attn_dropout, 
                 ff_dropout,
                 style='col'):
        
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed =  nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, mask = None):
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)
        _, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers: 
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        else:
             for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        return x


