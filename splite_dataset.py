import os
import shutil
import random


def count_files_in_dir(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])

def split_dataset(dataset_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        for class_name in ['yes', 'no']:
            if class_name == 'yes':
                for subfolder in ['Deep', 'Lobar', 'Subtentorial']:
                    os.makedirs(os.path.join(output_dir, split, class_name, subfolder), exist_ok=True)
            else:
                os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    # 遍历每个类别的文件夹
    for class_name in ['yes', 'no']:
        if class_name == 'yes':
            for subfolder in ['Deep', 'Lobar', 'Subtentorial']:
                subfolder_dir = os.path.join(dataset_dir, class_name, subfolder)
                files = os.listdir(subfolder_dir)
                random.shuffle(files)
                
                # 计算每个集合的样本数量
                total = len(files)
                train_count = int(total * train_ratio)
                val_count = int(total * val_ratio)
                test_count = total - train_count - val_count

                # 分割数据
                train_files = files[:train_count]
                val_files = files[train_count:train_count + val_count]
                test_files = files[train_count + val_count:]

                # 复制文件到相应的目录
                for f in train_files:
                    shutil.copy(os.path.join(subfolder_dir, f), os.path.join(output_dir, 'train', class_name, subfolder, f))
                for f in val_files:
                    shutil.copy(os.path.join(subfolder_dir, f), os.path.join(output_dir, 'val', class_name, subfolder, f))
                for f in test_files:
                    shutil.copy(os.path.join(subfolder_dir, f), os.path.join(output_dir, 'test', class_name, subfolder, f))
        else:
            class_dir = os.path.join(dataset_dir, class_name)
            files = os.listdir(class_dir)
            random.shuffle(files)
            
            # 计算每个集合的样本数量
            total = len(files)
            train_count = int(total * train_ratio)
            val_count = int(total * val_ratio)
            test_count = total - train_count - val_count

            # 分割数据
            train_files = files[:train_count]
            val_files = files[train_count:train_count + val_count]
            test_files = files[train_count + val_count:]

            # 复制文件到相应的目录
            for f in train_files:
                shutil.copy(os.path.join(class_dir, f), os.path.join(output_dir, 'train', class_name, f))
            for f in val_files:
                shutil.copy(os.path.join(class_dir, f), os.path.join(output_dir, 'val', class_name, f))
            for f in test_files:
                shutil.copy(os.path.join(class_dir, f), os.path.join(output_dir, 'test', class_name, f))



# 使用示例
dataset_dir = "dataset/"  # 数据集根目录，包含 'yes' 和 'no' 文件夹
output_dir = 'dataset2/'    # 输出目录，将保存划分后的数据集

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

split_dataset(dataset_dir, output_dir)

# 输出每个文件夹下的图片数量
for split in ['train', 'val', 'test']:
    for class_name in ['yes', 'no']:
        if class_name == 'yes':
            for subfolder in ['Deep', 'Lobar', 'Subtentorial']:
                subfolder_path = os.path.join(output_dir, split, class_name, subfolder)
                print(f'{split}/{class_name}/{subfolder}: {count_files_in_dir(subfolder_path)} images')
        else:
            class_path = os.path.join(output_dir, split, class_name)
            print(f'{split}/{class_name}: {count_files_in_dir(class_path)} images')
