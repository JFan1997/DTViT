from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models import alexnet
import torch.nn as nn
import torch


class DualAlexnet(nn.Module):
    def __init__(self, num_class1=2, num_class2=4, pretrained=True):
        super(DualAlexnet, self).__init__()
        self.backbone = alexnet(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier1 = nn.Linear(9216, num_class1)
        self.classifier2 = nn.Linear(9216, num_class2)


    def forward(self, x):
        x = self.encoder(x)
        x= x.view(x.size(0), -1)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2


if __name__ == "__main__":
    model = DualAlexnet()
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output[0].shape, output[1].shape)
    # model2 = nn.Sequential(*list(pretrained_net.children())[:-2])
    #
# model2 = nn.Sequential(*list(pretrained_net.children())[:-2])
