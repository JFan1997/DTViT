from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models import vgg16_bn,VGG16_BN_Weights,vgg11_bn,vgg16,vgg13_bn
import torch.nn as nn
import torch


class DualVgg16(nn.Module):
    def __init__(self, num_class1=2, num_class2=4, pretrained=True):
        super(DualVgg16, self).__init__()
        # self.backbone = vgg16(pretrained=pretrained)
        # self.backbone = vgg13_bn(pretrained=pretrained)
        # self.backbone = vgg16_bn(pretrained=pretrained)
        self.backbone = vgg16_bn(pretrained=pretrained)

        self.encoder = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier1 = nn.Sequential(
             nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_class1, bias=True),
        )
        self.classifier2 = nn.Sequential(
             nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_class2, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape,'encoder')
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2


if __name__ == "__main__":
    model = DualVgg16()
    # print(model)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output[0].shape, output[1].shape)
    # model2 = nn.Sequential(*list(pretrained_net.children())[:-2])
    #


# model2 = nn.Sequential(*list(pretrained_net.children())[:-2])
