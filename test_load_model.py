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
from models.vit import DualVisionTransformer
from torch.utils.tensorboard import SummaryWriter
from main import load_dataset,criterion1,criterion2
from plot_image import plot_image

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
    model_path = '/disk8t/jialiangfan/trained_models/{}'.format(model_name)
    test(model_path,model_type=args.model)