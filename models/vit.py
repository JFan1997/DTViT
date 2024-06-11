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



# class MLPBlock(nn.Module):
#     def __init__(self, dim, hidden_dim, out_dim, drop=0.):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, out_dim)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class ViTWithMLPHead(nn.Module):
    # def __init__(self, hidden_dim, mlp_dim=256, num_classes=2):
    #     super(ViTWithMLPHead, self).__init__()
    #     # 假设ViT的模型部分已经定义在这里或作为预训练模型加载
    #     self.mlp = nn.Sequential(
    #         nn.Linear(hidden_dim, mlp_dim),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Linear(mlp_dim, num_classes)
    #     )
        
    # def forward(self, x):
    #     # 通过MLP头部进行分类
    #     x = self.mlp(x)
    #     return x

class DualViT(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, num_class1,num_class2,patch=16, type='base', pretrained: bool = False,MLP=False):
        super(DualViT, self).__init__()
        if type == 'base':
            hidden_dim = 768
            if patch == 16:
                self.backbone = vit_b_16(pretrained=pretrained)
            elif patch == 32:
                self.backbone = vit_b_32(pretrained=pretrained)
        elif type == 'large':
            hidden_dim = 1024
            if patch == 16:
                self.backbone = vit_l_16(pretrained=pretrained)
            elif patch == 32:
                self.backbone = vit_l_32(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown ViT type {type}.")
        self.backbone.heads = nn.Identity()
        if MLP:
                self.head1=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, num_class1)
        )
                self.head2=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, num_class2)
        )
        else:
            self.head1 = nn.Linear(hidden_dim, num_class1)
            self.head2 = nn.Linear(hidden_dim, num_class2)


    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self.backbone(x)
        x1 = self.head1(x)
        x2 = self.head2(x)  
        return x1,x2

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = DualViT(2,4,patch=16, type='base', pretrained=False,MLP=True)
    print(model(input))
    model = DualViT(2,4,patch=32, type='base', pretrained=False)
    print(model(input))
    model = DualViT(2,4,patch=16, type='large', pretrained=False)
    print(model(input))
    model = DualViT(2,4,patch=32, type='large', pretrained=False)
    print(model(input))
    # model = DualViT(2,4,patch=16, type='huge', pretrained=False)
    # print(model(input))
