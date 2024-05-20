from torchvision import transforms,models,datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import os
from torch import nn
from torchvision.models import vit_b_16
import torch.optim as optim
from opt import read_args
import datetime
from five_dataset import MyDataset
from models.resnet18 import DualResNet
from models.vgg import DualVgg16
import time
from torch.utils.data import random_split
from plot_image import plot_image
from models.vit import DualVisionTransformer
from torch.utils.tensorboard import SummaryWriter
from main import load_dataset,criterion1,criterion2

def test(model_path,model_type):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    _,_,test_dataset=load_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    if model_type==0:
        model = DualVisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout=0.1,
            attention_dropout=0.1,
            num_classe1=2,
            num_classe2=4,
        )
    elif model_type==1:
        model = DualResNet(num_class1=2, num_class2=4)
    elif model_type==2:
        model = DualVgg16(num_class1=2, num_class2=4)
    model.load_state_dict(torch.load(model_path))


    # ##Testing
    # model.eval()
    # model.to(device)
    # start_time = time.time()
    # with torch.no_grad():
    #     running_loss = 0.
    #     running_corrects = 0
    #     for index, inputs  in enumerate(test_dataloader):
    #         image, labels1,labels2 = inputs['pixel_values'],inputs['label1'],inputs['label2']
    #         image = image.to(device)
    #         labels1= labels1.to(device)
    #         labels2= labels2.to(device)
    #         outputs1,outputs2  = model(image)
    #         _, preds1 = torch.max(outputs1, 1)
    #         _, preds2 = torch.max(outputs2, 1)
    #         # batch loss
    #         loss1 = criterion1(outputs1, labels1)
    #         loss2 = criterion2(outputs2, labels2)
    #         # loss = loss1 + loss2
    #         loss = loss1 + loss2
    #         # total loss
    #         running_loss += loss.item() * image.size(0)

    #         running_corrects += torch.sum(preds1 == labels1.data)
    #         running_corrects += torch.sum(preds2 == labels2.data)
    #         if index == 0:
    #             plot_image(image, labels1, labels2, preds1, preds2)
    #     epoch_loss = running_loss / len(test_dataset)
    #     epoch_acc = running_corrects / (2* len(test_dataset)) * 100.
    #     print('[Test] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.
    #         format(epoch_loss, epoch_acc, time.time() - start_time))


if __name__ == '__main__':
    args = read_args()
    print("model",args.model)
    model_name="dataset2-epoche_50-model_type_0-pretrained_True-argumentation_True-batch_size_32-optimizer_type-1.pth"
    model_name="dataset2-epoche_50-unfreeze_1-pretrained_True-argumentation_True-batch_size_32-optimizer_type-0.pth"
    model_name="dataset2-50-resnet_unfreeze_pretrained-True-32.pth"
    model_name="dataset2-epoche_50-model_type_2-pretrained_False-argumentation_True-batch_size_32-optimizer_type-0.pth"
    model_name="dataset2-50-resnet_unfreeze_pretrained-True-32.pth"
    
    # 
    model_name="dataset2-epoche_1-model_type_0-pretrained_False-argumentation_True-batch_size_32-optimizer_type-0.pth"
    model_name="dataset2-epoche_50-model_type_0-pretrained_True-argumentation_True-batch_size_32-optimizer_type-0.pth"
    model_path = '/disk2/fjl2401/trained_models/{}'.format(model_name)
    test(model_path,model_type=args.model)