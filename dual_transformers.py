
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize,
                                    RandomRotation,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    RandomAdjustSharpness,
                                    Resize, 
                                    ToTensor)

image_mean, image_std = processor.image_mean, processor.image_std
height = processor.size["height"]
width = processor.size["width"]
size = (height, width)
print("Size: ", size)
# Image mean:  [0.5, 0.5, 0.5]
# Image std:  [0.5, 0.5, 0.5]
print("Image mean: ", image_mean)
print("Image std: ", image_std)

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

def train_transforms(image):
        return _train_transforms(image)

def val_transforms(image):
        return _val_transforms(image)


from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

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



from torch.utils.data import random_split
data_dir='/home/jialiangfan/DTViT/dataset/'
dataset=MyDataset(data_dir)
train_size = int(0.8 * len(dataset))
val_size = int(0.1*len(dataset))
test_size=len(dataset) - train_size-val_size
train_dataset, val_dataset,test_dataset = random_split(dataset, [train_size,val_size,test_size])
# train_dataset = MyDataset(data_dir,test_frac=0.15,section="training",data_augmentation=True)
# test_dataset=MyDataset(data_dir,test_frac=0.15,section="test",data_augmentation=True)


len(train_dataset),len((test_dataset)),len(val_dataset)
# 155+98=253


# def collate_fn(examples):
#     pixel_values = torch.stack([example["pixel_values"] for example in examples])
#     labels = torch.tensor([example["label"] for example in examples])
#     return {"pixel_values": pixel_values, "labels": labels}



train_dataloader = DataLoader(train_dataset,batch_size=32)
test_dataloader = DataLoader(test_dataset,batch_size=4)


from transformers.models.vit.modeling_vit import * 

_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_CONFIG_FOR_DOC = "ViTConfig"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

class MLP_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DualViTForImageClassification(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels1 = 2
        self.num_labels2 = 4
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.classifier1 = MLP_Classifier(config.hidden_size, 512, 2)
        self.classifier2 = MLP_Classifier(config.hidden_size, 512, 4)

        # self.classifier2 = nn.Linear(config.hidden_size, 4) 
        # self.classifier1 = nn.Linear(config.hidden_size, 2) 


        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        label1: Optional[torch.Tensor] = None,
        label2: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # vit的输出是一个元组，第一个元素是最后一个token的输出，第二个元素是所有token的输出
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits1 = self.classifier1(sequence_output[:, 0, :])
        logits2 = self.classifier2(sequence_output[:, 0, :])
        loss = None
        loss_fct = CrossEntropyLoss()
        if label1 is not None:
            # print('this is label1',label1)
            # move labels to correct device to enable model parallelism
            label1 = label1.to(logits1.device)
            loss1 = loss_fct(logits1.view(-1, self.num_labels1), label1.view(-1))
        if label2 is not None:
            # move labels to correct device to enable model parallelism
            label2 = label2.to(logits2.device)
            # print('this is label2',label2)
            # print('this is logits2',logits2)
            loss2 = loss_fct(logits2.view(-1, self.num_labels2), label2.view(-1))
        loss=loss1+loss2
        if not return_dict:
            # print("return dict")
            output1 = (logits1,) + outputs[1:]
            output2 = (logits2,) + outputs[1:]
            return ((loss,) + output1) if loss is not None else output1, ((loss,) + output2) if loss is not None else output2
            # return ((loss,) + output) if loss is not None else output
        # print("return imageclassfieroutput")
        # 走的是这里
        return ImageClassifierOutput(
            loss=loss,
            logits=(logits1,logits2),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



from transformers import ViTConfig
from torch import nn
config=ViTConfig()
config.problem_type="single_label_classification"
# config.use_return_dict=True

import argparse

parser = argparse.ArgumentParser(description='Training Configuration')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model or not')

pretrained=parser.parse_args().pretrained
print("pretrained: ", pretrained)

if pretrained:
    model = DualViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",config=config)
else:
    model = DualViTForImageClassification(config=config)


# model.classifier=nn.Linear(in_features=768, out_features=train_dataset.num_class, bias=True)

from transformers import TrainingArguments, Trainer
import os

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

metric_name = "accuracy"
args = TrainingArguments(
    "transformers_models",
    evaluation_strategy="epoch",
    learning_rate=2e-5, #0.00002, #0.00002
    per_device_train_batch_size=32,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='transformers_models',
    save_strategy="epoch",
    remove_unused_columns=False)
# args.set_optimizer(name="sgd")
# Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, set the training and evaluation batch_sizes and customize the number of epochs for training, as well as the weight decay.
# We also define a `compute_metrics` function that will be used to compute metrics at evaluation. We use "accuracy" here.

from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # print('this is predictions',predictions, predictions.shape)
    # label是一个元祖，predict只返回了第一个分类器的结果
    predictions1,predictions2=predictions
    labels1,labels2=labels
    predictions1 = np.argmax(predictions1, axis=1)
    predictions2 = np.argmax(predictions2, axis=1)
    accuracy1=accuracy_score(predictions1, labels1)
    accuracy2=accuracy_score(predictions2, labels2)
    accuracy=(accuracy1+accuracy2)/2
    return dict(accuracy=accuracy)


import torch

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    
)

 

import os
trainer.train()




outputs = trainer.predict(test_dataset)

print(outputs.metrics)











