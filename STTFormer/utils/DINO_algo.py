import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from model.sttformer import Model

class DINO(nn.Module):
    def __init__(self,
                 backbone,
                 projection,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 bottleneck_channels):
        super().__init__()