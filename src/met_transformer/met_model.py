import torch
from torch import nn
from typing import List
from src.met_transformer.met_blocks import FeaturedWeightedEmbedding

class Stormer (nn.Module):

    def __init__(self,
                img_size: int,
                variables: List[str],
                patch_size: int = 2,
                hidden_size: int = 1024,
                depth: int = 24,
                num_heads: int = 16,
                mlp_ratio: float = 4.0    
                ):
        
        super().__init__()

        self.embedding = FeaturedWeightedEmbedding(
            feature_names=variables,
            embed_dim=1024, 
            init_weights=[0, 0],
            decay_rate=0.2
        )