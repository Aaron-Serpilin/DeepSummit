import torch
from torch import nn
from typing import List, Tuple, Dict
from src.met_transformer.met_blocks import FeaturedWeightedEmbedding, Block, FinalLayer
import math
import warnings

class Stormer (nn.Module):

    def __init__(self,
                img_size: Tuple[int, int],
                variables: List[str],
                met_weights: Dict[str, float],
                patch_size: int = 2,
                hidden_size: int = 1024,
                depth: int = 24,
                num_heads: int = 16,
                mlp_ratio: float = 4.0    
                ):
        
        super().__init__()

        if img_size[0] % patch_size != 0:
            pad_size = patch_size - img_size[0] % patch_size
            img_size = (img_size[0] + pad_size, img_size[1])

        self.img_size = img_size
        self.variables = variables
        self.patch_size = patch_size

        self.embedding = FeaturedWeightedEmbedding(
            feature_names=variables,
            embed_dim=hidden_size, 
            init_weights=met_weights,
            decay_rate=0.2
        )

        self.embed_norm_layer = nn.LayerNorm(hidden_size)

        # Creation of transformer blocks backbone
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # Prediction layer
        self.head = FinalLayer(hidden_size, patch_size, len(variables))

        self.initialize_weights()

    def initialize_weights (self):

        # Cut & Paste from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/weight_init.py?
        def trunc_normal (tensor: torch.Tensor,
                            mean: float = 0., 
                            std: float = 1., 
                            a: float = -2.,
                            b: float = 2.) -> torch.Tensor:
            
            """
            Fills `tensor` with values drawn from a truncated normal distribution.
            All ops happen inside torch.no_grad() so we don’t break the computational graph.
            """
            
            def norm_cdf(x):
                return (1. + math.erf(x / math.sqrt(2.))) / 2.
            
            if (mean < a - 2 * std) or (mean > b + 2 * std):
                warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                    "The distribution of values may be incorrect.",
                    stacklevel=2)
                
            # Causes all in-place ops on module.weight to not complain about mutating a leaf that requires grad
            with torch.no_grad():

                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)
                tensor.uniform_(2 * l - 1, 2 * u - 1)
                tensor.erfinv_()
                tensor.mul_(std * math.sqrt(2.))
                tensor.add_(mean)
                tensor.clamp_(min=a, max=b)
            return tensor


        def _basic_init (module):
            if isinstance(module, nn.Linear):
                trunc_normal(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)


    def unpatchify (self,
                    x: torch.Tensor,
                    h = None,
                    w = None):
        
        """
        x: (B, L, V * patch_size**2) → output (B, V, H, W)
        """
        
        p = self.patch_size
        v = len(self.variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1], "sequence length mismatch"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs
    
    def forward (self, 
                 x: torch.Tensor,
                 mask: torch.Tensor = None,
                 target: torch.Tensor = None,
                 window_mask: torch.Tensor = None):
        
        """
        x:            Tensor   (B, T+1, F)  from WeatherDataset, already has CLS row.
        mask:         Tensor   (B, T+1)     your day-masks 
        window_mask:  Tensor   (B, n_windows, T+1)  for intra-sample views
        """

        # Embeds all features
        x = self.embedding(x)
        x = self.embed_norm_layer(x)

        cls_token = x[:, 0]

        for block in self.blocks:
            x = block(x, cls_token)

        x = self.head(x, cls_token)
        x = self.unpatchify(x)

        return x