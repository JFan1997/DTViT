from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models import squeezenet1_0, densenet161
import torch.nn as nn
import torch


class DualSqueezeNet(nn.Module):
    def __init__(self, num_class1=2, num_class2=4, pretrained=True):
        super(DualSqueezeNet, self).__init__()
        self.backbone = squeezenet1_0(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_class1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_class2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        return x1, x2


if __name__ == "__main__":
    model = DualSqueezeNet()
    # print(model)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output[0].shape, output[1].shape)
    # model2 = nn.Sequential(*list(pretrained_net.children())[:-2])
    #


# model2 = nn.Sequential(*list(pretrained_net.children())[:-2])
