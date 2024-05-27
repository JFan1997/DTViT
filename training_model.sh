#!/bin/bash

# # resent18
# python main.py --data_argumentation True --optimizer_type 2 --model 1 --num_epochs 10 --device 1 --pretrained > ./log_no_pretrained/resnet18.log 2>&1

# # alexnet
# python main.py --data_argumentation True --optimizer_type 2 --model 3 --num_epochs 10 --device 1 --pretrained > ./log_no_pretrained/alexnet.log 2>&1

# # sqeezenet
# python main.py --data_argumentation True --optimizer_type 2 --model 4 --num_epochs 10 --device 1 --pretrained > ./log_no_pretrained/alexnet.log 2>&1

# # resnet34
# python main.py --data_argumentation True --optimizer_type 2 --model 5 --num_epochs 10 --device 1 --pretrained > ./log_no_pretrained/resnet34.log 2>&1

# # resnet50
# python main.py --data_argumentation True --optimizer_type 2 --model 6 --num_epochs 10 --device 1 --pretrained > ./log_no_pretrained/resnet50.log 2>&1

# # densenet
# python main.py --data_argumentation True --optimizer_type 2 --model 7 --num_epochs 10 --device 1 --pretrained > ./log_no_pretrained/densenet.log 2>&1

# # vit
# python main.py --data_argumentation True --optimizer_type 2 --model 0 --num_epochs 10 --device 1 --pretrained > ./log_no_pretrained/original_dtvit.log 2>&1

# # vit
# CUDA_VISIBLE_DEVICES=0  python dual_transformers.py --pretrained  > ./log_no_pretrained/dualViT.log 2>&1


# model 8
python main.py --data_argumentation True --optimizer_type 2 --model 8 --num_epochs 10 --device 1 --pretrained > ./logs_pretrained/vit_base_16.log 2>&1
python main.py --data_argumentation True --optimizer_type 2 --model 9 --num_epochs 10 --device 1 --pretrained > ./logs_pretrained/vit_base_32.log 2>&1
python main.py --data_argumentation True --optimizer_type 2 --model 10 --num_epochs 10 --device 1 --pretrained > ./logs_pretrained/vit_large_16.log 2>&1
python main.py --data_argumentation True --optimizer_type 2 --model 11 --num_epochs 10 --device 1 --pretrained > ./logs_pretrained/vit_large_32.log 2>&1
python main.py --data_argumentation True --optimizer_type 2 --model 12 --num_epochs 10 --device 1 --pretrained > ./logs_pretrained/vit_huge_14.log 2>&1
