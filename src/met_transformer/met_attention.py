import torch
from torch import nn
import torch.nn.functional as F

class Attention (nn.Module):

    """
    Multi-head self-attention block implemented in pure PyTorch based on Stormer's
    MemEffAttentionPT block. 
    
    Args:
        dim (int): Total feature dimension of the model.
        heads (int): Number of attention heads.
        dropout (float): Dropout probability for attention weights and output projection.
    """

    def __init__(self,
                 dim: int,
                 heads: int = 8,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0
                 ) -> None:
        
        super().__init__()
        assert dim % heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = heads
        self.head_dim =  dim // heads
        self.scale =self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.attn_drop = nn.Dropout(attn_drop)
        # Output projection and dropout
        self.proj = nn.Linear(dim, dim, bias = True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
       
        # Input has shape (Batch size, Sequence length, Dimension)
        B, T, D = x.shape

        qkv = self.qkv(x) # self.qkv: D -> 3 * D, so qkv = (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim) # reshape into (B, T, 3, H, head_dim)
        q, k, v = qkv.unbind(dim=2) # splitting the 3-component vector axis

        # Reordering to do attention per head -> (B, H, T, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # raw attention (dot_product) scores

        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        attn_probs = F.softmax(attn_scores, dim=1) # probabilities
        attn_probs = self.attn_drop(attn_probs) # dropping some links for regularization

        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D)

        # Final linear projection back to D dimensions
        out = self.proj(attn_out)
        out = self.proj_drop(out)
        return out