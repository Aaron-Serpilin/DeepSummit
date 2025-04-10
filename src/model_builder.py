import torch
from torch import nn

class TabularCNN(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_units: int,
                 batch_size: int,
                 channel_input: int,
                 dropout: float) -> None:
                super().__init__()

                self.conv_block_1 = nn.Sequential()
                self.conv_block_2 = nn.Sequential()
                self.conv_block_3 = nn.Sequential()
                self.classifier = nn.Sequential()

    def forward (self, x: torch.Tensor):
            return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))) # <- leverage the benefits of operator fusion
            