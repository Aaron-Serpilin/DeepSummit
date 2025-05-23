import torch
from torch import nn
from src.met_transformer.met_attention import Attention
from typing import List, Dict
import re

### Helpers ###

def modulate (x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

### Blocks ##3

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
    
class MLP (nn.Module):

    """
    Variation of the MLP block from the TabularDataset. The main modification is the incorporation of Swish
    and GLU to tackle the negative values that can arise in the weather data. 

    Args:
        dims (List[int]): List of layer sizes, e.g. [in_dim, hidden_dim, out_dim]
    """

    def __init__ (self,
                  dims,
                  act = None
    ):
        super().__init__()
        layers: list[nn.Module] = []
        # Except for the final layer, we project each hidden layer to 2x its width for a GLU split
        # We then apply Swish immediately to retain smooth gradients on negatives
        for i in range(len(dims) - 1):
            dim_in, dim_out = dims[i], dims[i+1]

            if i < len(dims) - 2:
                layers.append(nn.Linear(dim_in, dim_out * 2))
                layers.append(GLU(dim=-1))
                act = default(act, Swish())
                layers.append(act)
            else: 
                layers.append(nn.Linear(dim_in, dim_out))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
        
class Block (nn.Module):

    """
    Stormer Transformer block with adaptive LayerNorm and gated residuals.

    Args:
        hidden_size (int): embedding dimension for tokens.
        num_heads (int): number of attention heads.
        mlp_ratio (float): factor to scale hidden_size for MLP's hidden dimension.
        **block_kwargs: additional keyword args for the Attention module.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio=4.0,
                 **block_kwargs):
        super().__init__()

        # Pre-attention norm
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Attention Wrapper
        self.attn = Attention(dim=hidden_size, heads=num_heads, attn_drop=0.1, proj_drop=0.1)
        # Pre-MLP norm
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    
        # Two-layer MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = MLP([hidden_size, mlp_hidden, hidden_size])
        # Adaptive LayerNorm modulation for MSA and MLP
        self.adaLN_modulation = nn.Sequential(
            Swish(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention sub layer
        y = self.norm1(x)
        y = modulate(y, shift_msa, scale_msa)
        y = self.attn(y)
        x = x + gate_msa.unsqueeze(1) * y

        # MLP sub layer
        z = self.norm2(x)
        z = modulate(z, shift_mlp, scale_mlp)
        z = self.mlp(z)
        x = x + gate_mlp.unsqueeze(1) * z
        
        return x
    
class FinalLayer (nn.Module):

    """
    Final reconstruction layer with adaptive LayerNorm modulation.

    Args:
        hidden_size (int): dimension of hidden tokens.
        patch_size (int): spatial patch dimension.
        out_channels (int): number of output channels per patch.
    """

    def __init__ (self,
                  hidden_size: int,
                  patch_size: int,
                  out_channels: int
                  ):

        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            Swish(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class FeaturedWeightedEmbedding (nn.Module):

    """
    Embeds a sequence of feature vectors with learnable or fixed feature-wise weights.
    Amongst the list of weather variables, some have a more significant role in determining summit success.
    Therefore, said features are assigned a more dominant weight while still accounting for all factors. 

    Furthermore, the class also applies a temporal decay based on each feature's day-offset suffix
    as the weather of the more proximate days has a greater impact on the weather of the predicted day. 

    Args:
        feature_names (List[str]): Names of each input feature (length = num_features).
        embed_dim (int): Dimension of the output embeddings.
        init_weights ([Dict[str, float]): Optional mapping from feature_name to initial weight.
            Features not in this dict default to weight 1.0.
        decay_rate (float): Exponential decay rate per day offset; weight *= exp(-decay_rate * offset).
    """

    def __init__(self,
                 feature_names: List[str],
                 embed_dim: int,
                 init_weights: Dict[str, float] = None,
                 decay_rate: float = 0.2
                 ):
        
        super().__init__()
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.embed_dim = embed_dim
        weight_tensor = torch.ones(self.num_features, dtype=torch.float32)

        # Initial base weights
        if init_weights:
            for i, name in enumerate(feature_names):
                if name in init_weights:
                    weight_tensor[i] = init_weights[name]

        # Temporal decay based on day-offset in the feature name
        offsets = []
        pattern = re.compile(r"_t-(\d+)$")

        for name in feature_names:
            m = pattern.search(name)
            offset = int(m.group(1)) if m else 0
            offsets.append(offset)
        offsets = torch.tensor(offsets, dtype=torch.float32)
        decay = torch.exp(-decay_rate * offsets)
        weight_tensor *= decay

        # We initialize learnable per-feature weights
        self.feature_weights = nn.Parameter(weight_tensor, requires_grad=True)     

        # Linear projection from weighted features to embedding dimension
        self.proj = nn.Linear(self.num_features, embed_dim)

    def forward (self,
                 x: torch.Tensor
                 ) -> torch.Tensor:
        
        # Scale features by learned weights
        x_weighted = x * self.feature_weights.unsqueeze(0).unsqueeze(0)

        # Project to embedding space
        embeddings = self.proj(x_weighted)
        return embeddings
