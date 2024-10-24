from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models import resnet18,resnet34,resnet50 
import torch.nn as nn
import torch

class DualResNet(nn.Module):
    def __init__(self, num_class1=2, num_class2=3,pretrained=True,layer=18):
        super(DualResNet, self).__init__()

        if layer==18:
            backbone=resnet18(pretrained=pretrained)
            self.fc1=nn.Linear(512, num_class1)
            self.fc2=nn.Linear(512, num_class2)
        elif layer==34:
            backbone=resnet34(pretrained=pretrained)
            self.fc1=nn.Linear(512, num_class1)
            self.fc2=nn.Linear(512, num_class2)
        elif layer==50:
            backbone=resnet50(pretrained=pretrained)
            self.fc1=nn.Linear(2048, num_class1)
            self.fc2=nn.Linear(2048, num_class2)
        self.encoder=nn.Sequential(*list(backbone.children())[:-1])

        self.dropout=nn.Dropout(0.5)
    def forward(self, x):
        x=self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.dropout(x)
        # print(x.shape,'encoder')
        x1=self.fc1(x)
        x2=self.fc2(x)
        return x1, x2
    

if __name__ == "__main__":
    model = DualResNet(layer=50)
    # print(model)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output[0].shape, output[1].shape)

