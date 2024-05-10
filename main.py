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
from models.vit import DualVisionTransformer
from models.alexnet import DualAlexnet
from models.resnet18 import  DualResNet
from models.squeezenet import DualSqueezeNet
from models.densenet import DualDensenet
from models.vgg import DualVgg16
import time
from torch.utils.data import random_split
from plot_image import plot_image
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize,
                                    RandomRotation,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    RandomAdjustSharpness,
                                    Resize, 
                                    ToTensor)

criterion1 = nn.CrossEntropyLoss()  #(set loss function)
criterion2 = nn.CrossEntropyLoss()  #(set loss function)


def load_dataset(data_argumentation=False): 
    data_dir='/home/jialiangfan/head_blood/dataset'
    dataset=MyDataset(data_dir,balance=data_argumentation)
    train_size = int(0.8 * len(dataset))
     
    val_size = int(0.1*len(dataset))
    test_size=len(dataset) - train_size-val_size
    train_dataset, val_dataset,test_dataset = random_split(dataset, [train_size,val_size,test_size])
    # train_dataset = MyDataset(data_dir,test_frac=0.15,section="training")
    # test_dataset=MyDataset(data_dir,test_frac=0.15,section="test")
    return train_dataset,val_dataset,test_dataset


def train(num_epochs = 50, data_argumentation=False,batch_size=8,model_type=0,pretrained=True,optimizer_type=0,device=2):
    device = torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
    experiment_name="dataset-epoche_{}-model_type_{}-pretrained_{}-argumentation_{}-batch_size_{}-optimizer_type-{}".format(num_epochs, model_type, pretrained,data_argumentation,batch_size,optimizer_type)
    writer = SummaryWriter('./runs/{}/'.format(experiment_name))
    train_dataset,val_dataset,_=load_dataset(data_argumentation=data_argumentation)
    print("train_dataset",len(train_dataset))
    print("val_dataset",len(val_dataset))
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size)
    if model_type == 0:
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
                representation_size=768,
                pretrained=pretrained
            )
    elif model_type == 1:
        model= DualResNet(num_class1=2,num_class2=4,pretrained=pretrained)
    elif model_type == 2:
        model=DualVgg16(num_class1=2,num_class2=4,pretrained=pretrained)
    elif model_type == 3:
        model=DualAlexnet(num_class1=2,num_class2=4,pretrained=pretrained)
    elif model_type == 4:
        model=DualSqueezeNet(num_class1=2,num_class2=4,pretrained=pretrained)
    elif model_type == 5:
        model=DualResNet(num_class1=2,num_class2=4,pretrained=pretrained,layer=34)
    elif model_type == 6:
        model=DualResNet(num_class1=2,num_class2=4,pretrained=pretrained,layer=50)
    elif model_type == 7:
        model=DualDensenet(num_class1=2,num_class2=4,pretrained=pretrained)
    model.to(device)
    if optimizer_type == 0:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_type == 1:
        optimizer = optim.Adam(model.parameters(), lr=0.1)
    start_time = time.time() #(for showing time)
    for epoch in range(num_epochs): #(loop for every epoch)
        print("Epoch {} running".format(epoch)) #(printing message)
        """ Training Phase """
        model.train()    #(training model)
        running_loss = 0 #(set loss 0)
        running_corrects = 0 
        # load a batch data of images
        for i, inputs in enumerate(train_dataloader):
            image=inputs['pixel_values']
            labels1=inputs['label1']
            labels2=inputs['label2']
            # move to GPU
            image = image.to(device)
            labels1 = labels1.to(device) 
            labels2 = labels2.to(device) 
            # forward inputs and get output
            optimizer.zero_grad()
            pre_labels1,pre_labels2 = model(image)
            _, preds1 = torch.max(pre_labels1, 1)
            _, preds2 = torch.max(pre_labels2, 1)
            loss1 = criterion1(pre_labels1,labels1)
            loss2=criterion2(pre_labels2,labels2)
            loss=loss1+loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds1 == labels1.data)
            running_corrects += torch.sum(preds2 == labels2.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / (len(train_dataset)*2) * 100.
        current_time = datetime.datetime.now()
        # writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_acc, epoch)
        print('Current time: {} [Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(current_time,epoch, epoch_loss, epoch_acc, time.time() -start_time))
        
        # """ valing Phase """
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0
            for index, inputs  in enumerate(val_dataloader):
                image, labels1,labels2 = inputs['pixel_values'],inputs['label1'],inputs['label2']
                image = image.to(device)
                labels1= labels1.to(device)
                labels2= labels2.to(device)
                outputs1,outputs2 = model(image)
                _, preds1 = torch.max(outputs1, 1)
                _, preds2 = torch.max(outputs2, 1)
    #
                loss1 = criterion1(outputs1, labels1)
                loss2 = criterion2(outputs2, labels2)
                loss = loss1 + loss2
                # 迄今为止的loss
                running_loss += loss.item() * image.size(0)

                running_corrects += torch.sum(preds1 == labels1.data)
                running_corrects += torch.sum(preds2 == labels2.data)
            # 当前dataset的平均loss for each item
            epoch_loss = running_loss / len(val_dataset)
            epoch_acc = running_corrects / (2*len(val_dataset))* 100
            writer.add_scalar('Validation Loss', epoch_loss, epoch)
            writer.add_scalar('Validation Accuracy', epoch_acc, epoch)
            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time()- start_time))
    save_path = '/disk2/jialiangfan/trained_models/{}.pth'.format(experiment_name)
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    args = read_args()
    arugumentation = args.data_argumentation
    print("data_argument",arugumentation)
    print("num_epochs",args.num_epochs)
    print("batch_size",args.batch_size)
    print("model",args.model)
    print("pretrained",args.pretrained)
    print("optimizer_type",args.optimizer_type)
    print("device",args.device)

    train(num_epochs=args.num_epochs, data_argumentation= args.data_argumentation,batch_size=args.batch_size,
          model_type=args.model,pretrained=args.pretrained,optimizer_type=args.optimizer_type,device=args.device)
