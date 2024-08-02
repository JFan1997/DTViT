
from transformers import ViTImageProcessor

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
import torch

from sklearn.metrics import accuracy_score
import numpy as np
import torch
import os
from transformers import TrainingArguments, Trainer


from transformers import ViTForImageClassification,ViTConfig
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch
from PIL import Image
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize,
                                    RandomRotation,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    RandomAdjustSharpness,
                                    Resize, 
                                    ToTensor)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

image_mean, image_std = processor.image_mean, processor.image_std
height = processor.size["height"]
width = processor.size["width"]
size = (height, width)

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            Resize(size),
            RandomRotation(15),
            RandomAdjustSharpness(2),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
        # return _train_transforms(image)
    for item in examples:
        item['pixel_values'] = _train_transforms(item['image'])
    return examples

def val_transforms(examples):
    for item in examples:
        item['pixel_values'] = _val_transforms(item['image'])
    return examples


class MyDataset(Dataset):
    def __init__(self, data_dir,test_frac=0.15,section="training",data_augmentation=False):
        self.num_class = 0
        self.test_frac = test_frac
        self.data_augmentation=data_augmentation
        self.section=section
        self.transform=train_transforms if self.section=="training" else val_transforms
        self.generate_data_list(data_dir)


    def __len__(self):
        return len(self.data)
    
    def generate_data_list(self,data_dir):
        # 类别名 [yes,no]
        class_names = sorted(f"{x.name}" for x in Path(data_dir).iterdir() if x.is_dir())  # folder name as the class name
        # 2
        self.num_class = len(class_names)
        image_files_list = []
        image_class = []
        # [[class1图片列表][class2图片列表]]
        image_files = [[f"{x}" for x in (Path(data_dir) / class_names[i]).iterdir()] for i in range(self.num_class)]
        num_each = [len(image_files[i]) for i in range(self.num_class)]
        # 
        max_value=max(num_each)
        enlarge_factor=[max_value//num_each[i] for i in range(self.num_class)]
        if not self.data_augmentation:
            enlarge_factor=[1]*self.num_class
        print('this is the enlarge factor',enlarge_factor)

        class_name = []
        # 读取所有图片为一个二维list [[class1图片列表][class2图片列表]]
        # 对于每一类图片
        for i in range(self.num_class):
            # 将图片列表合并 [[class1图片列表][class2图片列表]] -> [class1图片列表+class2图片列表]
            image_files_list.extend(image_files[i]*enlarge_factor[i])
            # 为每个图片标记类别，类别标签从0开始，记录index [0,0,0,1,1,1]
            image_class.extend([i] * num_each[i]*enlarge_factor[i])
            # 为每个图片标记类别名 [yes,yes,yes,no,no,no]
            class_name.extend([class_names[i]] * num_each[i]*enlarge_factor[i])
        length = len(image_files_list)
        # 生成图片索引 [0,1,2,3,4,5]
        indices = np.arange(length)
        # 打乱图片顺序
        np.random.shuffle(indices)
        test_length = int(length * self.test_frac)
        if self.section == "test":
            section_indices = indices[:test_length]
        elif self.section == "training":
            section_indices = indices[test_length:]
        else:
            raise ValueError(
                f'Unsupported section: {self.section}, available options are ["training", "validation", "test"].'
            )
        def convert_image(image_path):
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image=image.convert('RGB')
            return image
        self.data=[{"image":convert_image(image_files_list[i]),"label": image_class[i]}  for i in section_indices ]
        self.data=self.transform(self.data)
    
    def __getitem__(self, index):
        # return self.data[index]
        img=self.data[index]["pixel_values"]
        label=self.data[index]['label']
        return {'pixel_values':img,'label':label}
        # return img,label


data_dir='/home/jialiangfan/DTViT/dataset1/'

train_dataset = MyDataset(data_dir,test_frac=0.15,section="training",data_augmentation=True)
test_dataset=MyDataset(data_dir,test_frac=0.15,section="test",data_augmentation=True)



# def collate_fn(examples):
#     pixel_values = torch.stack([example["pixel_values"] for example in examples])
#     labels = torch.tensor([example["label"] for example in examples])
#     return {"pixel_values": pixel_values, "labels": labels}




train_dataloader = DataLoader(train_dataset,batch_size=32)
test_dataloader = DataLoader(test_dataset,batch_size=4)


batch = next(iter(train_dataloader))



batch = next(iter(test_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)



config=ViTConfig()
config.num_labels=4
config.problem_type="single_label_classification"
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",config=config)
print(model.classifier)



metric_name = "accuracy"
args = TrainingArguments(
    "Brain-Tumor-Detection",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5, #0.00002, #0.00002
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
    # report_to="tensorboard",
)
# args.set_optimizer(name="sgd")



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))



trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    
)

trainer.train()

outputs = trainer.predict(test_dataset)
print(outputs.metrics)

