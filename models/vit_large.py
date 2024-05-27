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
from torchvision.models import vit_b_16,vit_l_16,vit_l_32


class DualViT_Large(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, num_class1,num_class2, patch=16, pretrained: bool = False,):
        super(DualViT_Large, self).__init__()
        if patch == 16:
            self.backbone = vit_l_16(pretrained=pretrained)
        elif patch == 32:
            self.backbone = vit_l_32(pretrained=pretrained)
        self.backbone.heads = nn.Identity()
        self.head1=nn.Linear(1024, num_class1)
        self.head2=nn.Linear(1024, num_class2)

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self.backbone(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        return x1,x2
    
    
if __name__ == '__main__':
    model = DualViT_Large(4)
    input=torch.randn(1,3,224,224)
    # print(model(input))    