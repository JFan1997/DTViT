from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional
import torch
from torch import nn
import math
from torch.nn import functional as F
from torchvision.models.vision_transformer import _log_api_usage_once, ConvStemConfig,MLPBlock
from torchsummary import summary
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models import vit_b_16


class ViT(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, num_class):
        super(ViT, self).__init__()
        self.backbone=vit_b_16(pretrained=False)
        self.backbone.heads = nn.Linear(in_features=768, out_features=num_class)

        

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        return self.backbone(x)
    
    
if __name__ == '__main__':
    model = ViT(4)
    input=torch.randn(1,3,224,224)
    print(model(input).shape)    