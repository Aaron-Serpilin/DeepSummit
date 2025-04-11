import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

WEIGHT_DECAY = 0.1
BATCH_SIZE = 32 # 256 is the original but due to hardware limitations we will lower it
LR = 0.0001
DROPOUT = 0.1

### Helpers ###

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

### Model ###


