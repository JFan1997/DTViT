
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
import matplotlib.pyplot as plt


# class MyDataset(Dataset):
#     def __init__(self, data_dir,section="train",balance=False):
#         self.num_class = 0
#         self.test_frac = test_frac
#         self.section=section
#         # self.transform=train_transforms if self.section=="training" else val_transforms
#         self.transform=train_transforms
#         if self.section=="train":
#             self.generate_data_list(data_dir+"/train")
#         elif self.section=="val":
#             self.generate_data_list(data_dir+"/val")
#         elif self.section=="test":
#             self.generate_data_list(data_dir+"/test")
#         if balance:
#             self.balance_classes()


from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize,
                                    RandomRotation,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    RandomAdjustSharpness,
                                    Resize, 
                                    ToTensor)

# image_mean, image_std = processor.image_mean, processor.image_std
# height = processor.size["height"]
# width = processor.size["width"]
# size = (height, width)
# print("Size: ", size)
# print("Image mean: ", image_mean)
# print("Image std: ", image_std)


# Image mean:  [0.5, 0.5, 0.5]
# Image std:  [0.5, 0.5, 0.5]
normalize = Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
# 

train_transforms = transforms.Compose([
    Resize(256),
    transforms.CenterCrop(224),
    RandomRotation(15),
    RandomAdjustSharpness(2),
    ToTensor(),
    normalize,
])

val_transforms = transforms.Compose([
    Resize(256),
    transforms.CenterCrop(224),
            ToTensor(),
            normalize,
])
# def train_transforms(image):
#     _train_transforms = Compose(
#         [
#             Resize(256),
#             transforms.CenterCrop(224), # 然后进行中心裁剪到模型期望的尺寸
#             RandomRotation(15),
#             RandomAdjustSharpness(2),
#             ToTensor(),
#             normalize,
#         ]
#     )

#     return _train_transforms(image)


# def val_transforms(image):
#     _val_transforms = Compose(
#         [
#             Resize(256),
#             transforms.CenterCrop(224), # 然后进行中心裁剪到模型期望的尺寸
#             ToTensor(),
#             normalize,
#         ]
#     )
#     return _val_transforms(image)


def generate_data_list(data_dir,balance=False):
    # 类别名 [yes,no]
    # class_names = sorted(f"{x.name}" for x in Path(data_dir).iterdir() if x.is_dir())  # folder name as the class name
    no_tumor_dir = os.path.join(data_dir, 'no')
    no_tumor_images = [(os.path.join(no_tumor_dir, img), 0, 3) for img in os.listdir(no_tumor_dir)]
    yes_tumor_dir = os.path.join(data_dir, 'yes')
    tumor_classes = {'Deep': 0, 'Lobar': 1, 'Subtentorial': 2}
    yes_tumor_images = []
    for tumor_class, label in tumor_classes.items():
        class_dir = os.path.join(yes_tumor_dir, tumor_class)
        yes_tumor_images += [(os.path.join(class_dir, img), 1, label) for img in os.listdir(class_dir)]
    samples = no_tumor_images + yes_tumor_images
    if balance:
        samples = balance_classes(samples)
    return samples


def balance_classes(samples):
    from collections import Counter
    # 3 暂时不分类
    label2_counter=Counter(x[2] for x in samples)
    print("this is label2 counter: ", label2_counter)
    max_label2_count = max(label2_counter.values())
    print("max label2 count: ", max_label2_count)
    # before balance
    print("total num before first balance: ", len(samples))
    print("Deep: ", label2_counter[0])
    print("Lobar: ", label2_counter[1])
    print("Subtentorial: ", label2_counter[2])
    # # 为了平衡类别，复制少数类的数据，先复制label2
    balanced_samples = []
    for key in label2_counter.keys():
        factor = max_label2_count // label2_counter[key]
        if key==3:
            factor=1
        for i in range(len(samples)):
            if samples[i][2] == key:
                balanced_samples.extend([samples[i]]*factor)
    
    samples = balanced_samples
    print("total num after first balance: ", len(samples))
    label2_counter=Counter(x[2] for x in samples)
    print("counter after first balance: ", label2_counter)
    balanced_samples=[]
    label1_counter = Counter(x[1] for x in samples)
    print("total num before second balance: ", len(samples))
    print("no tumor: ", label1_counter[0])
    print("tumor: ", label1_counter[1])
    max_label1_count = max(label1_counter.values())
    print("max label1 count: ", max_label1_count)
    for label in label1_counter.keys():
        factor = max_label1_count // label1_counter[label]
    #     # 复制因子次每个类别的数据
        for i in range(len(samples)):
            if samples[i][1] == label:
                balanced_samples.extend([samples[i]]* factor)
    samples = balanced_samples
    print("total num after balance: ", len(samples))
    label1_counter = Counter(x[1] for x in samples)
    print("no tumor: ", label1_counter[0])
    print("tumor: ", label1_counter[1])
    return samples



class MyDataset(Dataset):
    def __init__(self, data_list, section="train",balance=False):
        self.num_class = 0
        self.section=section
        self.samples =data_list
        if balance:
            self.samples = balance_classes(self.samples)
        # self.transform=train_transforms
        if self.section=="train":
            self.transform=train_transforms
        else:
             self.transform=val_transforms

    def __len__(self):
        return len(self.samples)
        # self.data=self.transform(self.data)
    
    def __getitem__(self, index):
        img_path, has_tumor, tumor_type = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        image=self.transform(image)
        #filename=self.data[index]['file_name']
        # # return {'pixel_values':img,'label':label}
        # return img,label,filename
        return {'pixel_values':image,'label1':has_tumor,'label2':tumor_type}
        # return image,has_tumor, tumor_type

    def visual_transform(self, image):
        first=transforms.Resize(256, image)
        second=transforms.CenterCrop(224, first)
        return first, second

if __name__ == "__main__":

    data_dir = './dataset'
    # data_list=generate_data_list(data_dir,balance=False)
    # train_size=int(0.8*len(data_list))
    # val_size=int(0.1*len(data_list))
    # test_size=len(data_list)-train_size-val_size
    # dataset = MyDataset(data_list,balance=False)
    # image=dataset[0]
    # print(image['pixel_values'].shape)
    # data = np.transpose(image['pixel_values'], (1, 2, 0))

    # plt.imshow(data)

    # plt.imshow(image['pixel_values'])
    # plot_image(image['pixel_values'])
