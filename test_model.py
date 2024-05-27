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
from torch.utils.tensorboard import SummaryWriter
from main import load_dataset,criterion1,criterion2,select_model



device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_,_,test_dataset=load_dataset(data_argumentation=True)
test_dataloader = DataLoader(test_dataset,batch_size=4)


def test(model_path,model_type):

    model=select_model(model_type,False)    
    model.load_state_dict(torch.load(model_path))


    ##Testing
    model.eval()
    model.to(device)
    start_time = time.time()
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for index, inputs  in enumerate(test_dataloader):
            image, labels1,labels2 = inputs['pixel_values'],inputs['label1'],inputs['label2']
            image = image.to(device)
            labels1= labels1.to(device)
            labels2= labels2.to(device)
            outputs1,outputs2  = model(image)
            _, preds1 = torch.max(outputs1, 1)
            _, preds2 = torch.max(outputs2, 1)
            # batch loss
            loss1 = criterion1(outputs1, labels1)
            loss2 = criterion2(outputs2, labels2)
            # loss = loss1 + loss2
            loss = loss1 + loss2
            # total loss
            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds1 == labels1.data)
            running_corrects += torch.sum(preds2 == labels2.data)
            if index == 0:
                plot_image(image, labels1.tolist(), labels2.tolist(), preds1.tolist(), preds2.tolist())
        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / (2* len(test_dataset)) * 100.
        print('[Test] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.
            format(epoch_loss, epoch_acc, time.time() - start_time))


if __name__ == '__main__':
    args = read_args()
    # model_name="dataset-epoche_10-model_type_0-pretrained_True-argumentation_True-batch_size_32-optimizer_type-2.pth"
    model_list=os.listdir('/disk8t/jialiangfan/trained_models/')
    
    for model in model_list:
        print('get the testing performance of model: ',model)
        model_type=int(model.split('_')[3].split('-')[0])
        print('model_type:',model_type)
        model_path = '/disk8t/jialiangfan/trained_models/{}'.format(model)
        try:
            test(model_path,model_type)
        except:
            print('error in model:',model)
            continue
        # test(model_path,model_type=model_type)