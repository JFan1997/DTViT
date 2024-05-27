
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


def train_transforms(image):
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

    return _train_transforms(image)
    # for item in examples:
        # item['pixel_values'] = _train_transforms(item['image'])
    # return examples

def val_transforms(examples):
    _val_transforms = Compose(
        [
            Resize(256),
            transforms.CenterCrop(224), # 然后进行中心裁剪到模型期望的尺寸
            ToTensor(),
            normalize,
        ]
    )
    for item in examples:
        item['pixel_values'] = _val_transforms(item['image'])
    return examples


class MyDataset(Dataset):
    def __init__(self, data_dir,test_frac=0.15,section="training",balance=False):
        self.num_class = 0
        self.test_frac = test_frac
        self.section=section
        self.transform=train_transforms if self.section=="training" else val_transforms
        self.generate_data_list(data_dir)
        if balance:
            self.balance_classes()


    def __len__(self):
        return len(self.samples)
    #
    def balance_classes(self):
        from collections import Counter
        # 3 暂时不分类
        label2_counter=Counter(x[2] for x in self.samples)
        print("this is label2 counter: ", label2_counter)
        max_label2_count = max(label2_counter.values())
        print("max label2 count: ", max_label2_count)
        # before balance
        print("total num before first balance: ", len(self.samples))
        print("Deep: ", label2_counter[0])
        print("Lobar: ", label2_counter[1])
        print("Subtentorial: ", label2_counter[2])
        # # 为了平衡类别，复制少数类的数据，先复制label2
        balanced_samples = []
        for key in label2_counter.keys():
            factor = max_label2_count // label2_counter[key]
            if key==3:
                factor=1
            for i in range(len(self.samples)):
                if self.samples[i][2] == key:
                    balanced_samples.extend([self.samples[i]]*factor)
        
        self.samples = balanced_samples
        print("total num after first balance: ", len(self.samples))
        label2_counter=Counter(x[2] for x in self.samples)
        print("counter after first balance: ", label2_counter)
        balanced_samples=[]
        label1_counter = Counter(x[1] for x in self.samples)
        print("total num before second balance: ", len(self.samples))
        print("no tumor: ", label1_counter[0])
        print("tumor: ", label1_counter[1])
        max_label1_count = max(label1_counter.values())
        print("max label1 count: ", max_label1_count)
        for label in label1_counter.keys():
            factor = max_label1_count // label1_counter[label]
        #     # 复制因子次每个类别的数据
            for i in range(len(self.samples)):
                if self.samples[i][1] == label:
                    balanced_samples.extend([self.samples[i]]* factor)
        self.samples = balanced_samples
        print("total num after balance: ", len(self.samples))
        label1_counter = Counter(x[1] for x in self.samples)
        print("no tumor: ", label1_counter[0])
        print("tumor: ", label1_counter[1])


    def generate_data_list(self,data_dir):
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
        self.samples = no_tumor_images + yes_tumor_images
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


if __name__ == "__main__":
    data_dir = '/home/jialiangfan/head_blood/dataset'
    dataset = MyDataset(data_dir,balance=True)