import torch
from torch import nn
from xformers.ops import memory_efficient_attention, unbind
from src.tab_transformer.tab_blocks import MLP
from src.met_transformer.met_attention import Attention

def modulate (x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Swish (nn.Module):

    """
    Swish is a smooth, non-monotonic function that arises as a slight modification of the sigmoid function.
    Swish's performance consistently matches or outperforms ReLU. 

    It is relevant for the meteorological data given the multiple negatives instances the features may have.
    ReLU outputs 0 for any negative value, leading to the "Dying ReLU" problem where it can output 
    0 for all the input values, consequently making the gradient 0, not allowing the gradients to be updated. 

    Swish maintains ReLU's behavior for positive values, while not outputting 0 for all the negative values. 

    swish(x) = x ⋅ σ(x)
    """

    def __init__ (self):
        super().__init__()

    def forward (self, x):
        return x * x.sigmoid()
    
class GLU (nn.Module):

    """
    Gating mechanism introduced in "Language Modeling with Gated Convolutional Networks".

    Allows the network to decide on a per-feature basis how much of the input should pass or be suppressed.

    The sigmoid computation of the gate tensor maps every value to (0, 1) where a value near 1 allows the 
    channel's contents to flow freely, while a near 0 value shuts if off. This allows outputs to be 
    dynamically scaled based on the gate rather than being entirely zeroed as in ReLU, which is key for us to
    appropriately analyze the negative weather values. 
    """
   
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        outputs, gate = x.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
        
class Block (nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio=4.0,
                 **block_kwargs):
        super().__init__()

        # Pre-attention norm
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Attention Wrapper
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # Pre-MLP norm
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Two-layer MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLP(dims=[hidden_size, mlp_hidden_dim, hidden_size], act=nn.GELU())

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention sublayer
        y = self.norm1(x)
        y = modulate(y, shift_msa, scale_msa)
        y = self.attn(y)
        x = x + gate_msa.unsqueeze(1) * y

        # MLP sublayer
        z = self.norm2(x)
        z = modulate(z, shift_mlp, scale_mlp)
        z = self.mlp(z)
        x = x + gate_mlp.unsqueeze(1) * z
        
        return x
