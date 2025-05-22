import torch
from torch import nn
from typing import List
from src.met_transformer.met_blocks import FeaturedWeightedEmbedding
from src.met_transformer.met_data_setup import weather_mapping

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

        # We initialize a higher weight for all wind, temperature, and pressure variables
        # Research suggest these variables are more significant in determining summit success
        met_weights = {
            key: 1.2 # an 20% boost to these key weights
            for key, description in weather_mapping.items()
            if any(term in description.lower() for term in ("wind", "temperature", "pressure"))
        }

        # We adapt the variable names of the weights to all the offset features
        offsets= range(0, 8)
        met_weights_with_offset = {
            f"{feat}_t-{off}":weight
            for feat, weight in met_weights.items()
            for off in offsets
        }
        ### Integrate regularization afterwards to avoid one feature dominating too much ###

        self.embedding = FeaturedWeightedEmbedding(
            feature_names=variables,
            embed_dim=1024, 
            init_weights=met_weights_with_offset,
            decay_rate=0.2
        )