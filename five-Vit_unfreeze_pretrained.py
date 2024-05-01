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
from models.transformer import DualVisionTransformer
from models.resnet import DualResNet
import time
from torch.utils.data import random_split
from plot_image import plot_image
from models.transformer import DualVisionTransformer
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize,
                                    RandomRotation,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    RandomAdjustSharpness,
                                    Resize, 
                                    ToTensor)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
normalize = Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
_train_transforms = Compose(
        [
            Resize(256),
            transforms.CenterCrop(224), # 然后进行中心裁剪到模型期望的尺寸
            RandomRotation(15),
            RandomAdjustSharpness(2),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(256),
            transforms.CenterCrop(224), # 然后进行中心裁剪到模型期望的尺寸
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(image):
    return _train_transforms(image)
    # for item in examples:
        # item['pixel_values'] = _train_transforms(item['image'])
    # return examples

def val_transforms(examples):
    for item in examples:
        item['pixel_values'] = _val_transforms(item['image'])
    return examples


criterion1 = nn.CrossEntropyLoss()  #(set loss function)
criterion2 = nn.CrossEntropyLoss()  #(set loss function)



def load_dataset(data_argumentation=False): 
    data_dir='/home/jialiangfan/head_blood/dataset2'
    dataset=MyDataset(data_dir,balance=data_argumentation)
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # train_dataset = MyDataset(data_dir,test_frac=0.15,section="training")
    # test_dataset=MyDataset(data_dir,test_frac=0.15,section="test")
    return train_dataset,test_dataset


def train(num_epochs = 50, data_argumentation=False,batch_size=8,model=0):
    train_dataset,test_dataset=load_dataset(data_argumentation=data_argumentation)
    print("train_dataset",len(train_dataset))
    print("test_dataset",len(test_dataset))
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size)
    if model == 0:
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
    elif model == 1:
        model= DualResNet(num_class1=2,num_class2=4)
    model.to(device)


    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
            # print("pre_labels",pre_labels1,pre_labels2)
            _, preds1 = torch.max(pre_labels1, 1)
            _, preds2 = torch.max(pre_labels2, 1)
            # print("type",preds1.dtype,labels1.dtype)
            # print("preds",preds1,preds2)
            # print("labels",labels1,labels2)
            loss1 = criterion1(pre_labels1,labels1)
            loss2=criterion2(pre_labels2,labels2)
            # 两个分类loss之和
            loss=loss1+loss2
            # print("loss",loss)
            # get loss value and update the network weights
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * image.size(0)
            # print("size",pre_labels1.size(),labels1.size())
            # print("data",labels1.data)
            # 两个分类的正确数
            running_corrects += torch.sum(preds1 == labels1.data)
            running_corrects += torch.sum(preds2 == labels2.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / (len(train_dataset)*2) * 100.
        current_time = datetime.datetime.now()

        print('Current time: {} [Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(current_time,epoch, epoch_loss, epoch_acc, time.time() -start_time))
        
        # """ Testing Phase """
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0
            for index, inputs  in enumerate(test_dataloader):
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
                running_loss += loss.item() * image.size(0)

                running_corrects += torch.sum(preds1 == labels1.data)
                running_corrects += torch.sum(preds2 == labels2.data)

            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects / (2*len(test_dataset))* 100
            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time()- start_time))
    save_path = 'dataset2-{}-vit_unfreeze_pretrained-{}-{}.pth'.format(num_epochs,data_argumentation,batch_size)
    torch.save(model.state_dict(), save_path)


def test():
    test_dataloader = DataLoader(test_dataset,batch_size=8)
    train_dataset,test_dataset=load_dataset()
    save_path = 'dataset2-50-vit_unfreeze_pretrained.pth'
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
    model.load_state_dict(torch.load(save_path))
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
            loss1 = criterion1(outputs1, labels1)
            loss2 = criterion2(outputs2, labels2)
            loss = loss1 + loss2
            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds1 == labels1.data)
            running_corrects += torch.sum(preds2 == labels2.data)
            if index == 0:
                plot_image(image, labels1, labels2)
        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / (2* len(test_dataset)) * 100.
        print('[Test] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.
            format(epoch_loss, epoch_acc, time.time() - start_time))


if __name__ == '__main__':
    args = read_args()
    arugumentation = args.data_argument
    print("data_argument",arugumentation)
    print("num_epochs",args.num_epochs)
    print("batch_size",args.batch_size)
    print("model",args.model)
    train(num_epochs=args.num_epochs, data_argumentation= args.data_argument,batch_size=args.batch_size,model=args.model)