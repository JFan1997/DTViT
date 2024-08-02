import argparse
import os
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from five_dataset import MyDataset, generate_data_list
from models.vit_base import DualVisionTransformer
from models.vit import DualViT
from models.alexnet import DualAlexnet
from models.resnet18 import DualResNet
from models.squeezenet import DualSqueezeNet
from models.densenet import DualDensenet
from models.vgg import DualVgg16
from models.vit_adapter import build_model
from opt import model_types, optimizer_types, device_types


def load_dataset(data_augmentation=False): 
    data_dir = '/home/jialiangfan/DTViT/dataset'
    data_list = generate_data_list(data_dir, balance=False)
    train_size = int(0.8 * len(data_list))
    val_size = int(0.1 * len(data_list))
    test_size = len(data_list) - train_size - val_size
    train_datalist, val_datalist, test_datalist = random_split(data_list, [train_size, val_size, test_size])
    
    train_dataset = MyDataset(train_datalist, balance=False, section='train')
    val_dataset = MyDataset(val_datalist, balance=False, section='val')
    test_dataset = MyDataset(test_datalist, balance=False, section='test')
    
    return train_dataset, val_dataset, test_dataset

def select_model(model_type, pretrained):
    model_classes = {
        0: DualVisionTransformer,
        1: DualResNet,
        2: DualVgg16,
        3: DualAlexnet,
        4: DualSqueezeNet,
        5: lambda: DualResNet(num_class1=2, num_class2=4, pretrained=pretrained, layer=34),
        6: lambda: DualResNet(num_class1=2, num_class2=4, pretrained=pretrained, layer=50),
        7: DualDensenet,
        8: DualViT,
        9: lambda: DualViT(num_class1=2, num_class2=4, patch=32, pretrained=pretrained),
        10: lambda: DualViT(num_class1=2, num_class2=4, type='large', patch=16, pretrained=pretrained),
        11: lambda: DualViT(num_class1=2, num_class2=4, type='large', patch=32, pretrained=pretrained),
        12: lambda: DualViT(num_class1=2, num_class2=4, type='huge', pretrained=pretrained),
        13: lambda: DualViT(num_class1=2, num_class2=4, type='base', pretrained=pretrained, MLP=True),
        14: lambda: DualViT(num_class1=2, num_class2=4, type='large', pretrained=pretrained, MLP=True),
        15: build_model
    }
    
    return model_classes[model_type]()


def get_optimizer(optimizer_type, model):
    if optimizer_type == 0:
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_type == 1:
        return optim.Adam(model.parameters(), lr=0.1)
    elif optimizer_type == 2:
        return optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)


def train(num_epochs=50, data_augmentation=False, batch_size=8, model_type=0, pretrained=True, optimizer_type=0, device=2):
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    experiment_name = f"dataset-epoch_{num_epochs}-model_type_{model_type}-pretrained_{pretrained}-augmentation_{data_augmentation}-batch_size_{batch_size}-optimizer_type-{optimizer_type}"
    writer = SummaryWriter(f'./runs/{experiment_name}/')
    
    train_dataset, val_dataset, test_dataset = load_dataset(data_augmentation=data_augmentation)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = select_model(model_type, pretrained)
    model.to(device)
    
    if model_type == 15:
        for n, value in model.image_encoder.named_parameters(): 
            value.requires_grad = "Adapter" in n
           
    optimizer = get_optimizer(optimizer_type, model)
    
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    best_model = None
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs in train_dataloader:
            images = inputs['pixel_values'].to(device)
            labels1 = inputs['label1'].to(device)
            labels2 = inputs['label2'].to(device)
            
            optimizer.zero_grad()
            outputs1, outputs2 = model(images)
            loss1 = criterion1(outputs1, labels1)
            loss2 = criterion2(outputs2, labels2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            
            _, preds1 = torch.max(outputs1, 1)
            _, preds2 = torch.max(outputs2, 1)
            
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds1 == labels1.data) + torch.sum(preds2 == labels2.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / (2 * len(train_dataset)) * 100
        
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_acc, epoch)
        
        print(f"Epoch [{epoch}/{num_epochs}] Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}% Time: {time.time() - start_time:.4f}s")
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs in val_dataloader:
                images = inputs['pixel_values'].to(device)
                labels1 = inputs['label1'].to(device)
                labels2 = inputs['label2'].to(device)
                
                outputs1, outputs2 = model(images)
                loss1 = criterion1(outputs1, labels1)
                loss2 = criterion2(outputs2, labels2)
                loss = loss1 + loss2
                
                _, preds1 = torch.max(outputs1, 1)
                _, preds2 = torch.max(outputs2, 1)
                
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds1 == labels1.data) + torch.sum(preds2 == labels2.data)
        
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_corrects.double() / (2 * len(val_dataset)) * 100
        
        writer.add_scalar('Validation Loss', epoch_val_loss, epoch)
        writer.add_scalar('Validation Accuracy', epoch_val_acc, epoch)
        
        print(f"Validation Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}%")
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model = model.state_dict()
            print(f"Best model found at epoch {epoch} with loss: {best_loss:.4f}")
    
    save_path = f'/disk8t/jialiangfan/trained_models/{experiment_name}.pth'
    torch.save(best_model, save_path)

if __name__ == '__main__':
    args = argparse.parse_args()
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Data augmentation: {args.data_augmentation}")
    print(f"Using pretrained model: {args.pretrained}")
    print(f"Optimizer type: {optimizer_types[args.optimizer_type]}")
    print(f"Device: {device_types[args.device]}")
    print(f"Model type: {model_types[args.model]}")
    train(num_epochs=args.num_epochs, data_argumentation=args.data_argumentation,batch_size=args.batch_size,
          model_type=args.model,pretrained=args.pretrained,optimizer_type=args.optimizer_type,device=args.device)
          