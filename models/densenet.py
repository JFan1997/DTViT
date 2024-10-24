from torchvision.models import densenet161
import torch.nn as nn
import torch


class DualDensenet(nn.Module):
    def __init__(self, num_class1=2, num_class2=4, pretrained=False):
        super(DualDensenet, self).__init__()
        self.backbone = densenet161(pretrained=pretrained)
        self.backbone.classifier=nn.Identity()
        self.classifier1 = nn.Linear(2208, num_class1)
        self.classifier2 = nn.Linear(2208, num_class2)


    def forward(self, x):
        x = self.backbone(x)
        # x= x.view(x.size(0), -1)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2

if __name__ == "__main__":
    model = DualDensenet()
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output[0].shape, output[1].shape)
    # model2 = nn.Sequential(*list(pretrained_net.children())[:-2])
    #
# model2 = nn.Sequential(*list(pretrained_net.children())[:-2])
