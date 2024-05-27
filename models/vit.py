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
from torchvision.models import vit_b_16, vit_l_16, vit_h_14, vit_b_32,vit_l_32,ViT_H_14_Weights

class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DualViT(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, num_class1,num_class2,patch=16, type='base', pretrained: bool = False):
        super(DualViT, self).__init__()
        if type == 'base':
            if patch == 16:
                self.backbone = vit_b_16(pretrained=pretrained)
            elif patch == 32:
                self.backbone = vit_b_32(pretrained=pretrained)
            
            self.backbone.heads = nn.Identity()
            self.head1=nn.Linear(768, num_class1)
            self.head2=nn.Linear(768, num_class2)
        elif type == 'large':
            if patch == 16:
                self.backbone = vit_l_16(pretrained=pretrained)
            elif patch == 32:
                self.backbone = vit_l_32(pretrained=pretrained)
            self.backbone.heads = nn.Identity()
            self.head1=nn.Linear(1024, num_class1)
            self.head2=nn.Linear(1024, num_class2)
        elif type == 'huge':
            self.backbone = vit_h_14(weights=ViT_H_14_Weights.DEFAULT)
            self.backbone.heads = nn.Identity()
            self.head1=nn.Linear(1280, num_class1)
            self.head2=nn.Linear(1280, num_class2)
        else:
            raise ValueError(f"Unknown ViT type {type}.")
 

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self.backbone(x)
        x1 = self.head1(x)
        x2 = self.head2(x)  
        return x1,x2

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = DualViT(2,4,patch=16, type='base', pretrained=False)
    print(model(input))
    model = DualViT(2,4,patch=32, type='base', pretrained=False)
    print(model(input))
    model = DualViT(2,4,patch=16, type='large', pretrained=False)
    print(model(input))
    model = DualViT(2,4,patch=32, type='large', pretrained=False)
    print(model(input))
    model = DualViT(2,4,patch=16, type='huge', pretrained=False)
    print(model(input))
