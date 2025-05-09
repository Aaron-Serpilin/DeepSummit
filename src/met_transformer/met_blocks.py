import torch
from torch import nn

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