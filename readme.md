## 数据预处理


## 数据集

1. 原始数据集：/disk8t/jialiangfan/trained_models/dataset/medical_data/2018,2019,2020,2021 DCM文件
2. 转为png数据集：/disk8t/jialiangfan/trained_models/dataset/medical_data/medical_images/2018,2019,2020,2021, jpg文件
3. /disk8t/jialiangfan/trained_models/dataset/medical_data/images: 将所有的病人数据大杂烩，放在一起
4. 每个病人挑一张最好的，/home/jialiangfan/head_blood/dataset/no，没有脑出血的，总共199张
5. 每个病人挑一张最好的，有脑出血的，/home/jialiangfan/head_blood/dataset/yes，每个人两张，608张


## 运行指令

## using adam as the optimizer 

### 测试vit模型
有数据增强：nohup python main.py --data_argument True --batch_size 32 --num_epochs 50 
没数据增强：nohup python main.py --data_argument False --batch_size 8 --num_epochs 50

## 训练resnet模型
nohup python main.py --data_argument True --batch_size 32 --num_epochs 50 --model 1

## 查看log日志

tensorboard --logdir=runs --port=8080

### 测试


