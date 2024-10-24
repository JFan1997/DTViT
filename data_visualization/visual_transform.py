import cv2

image='./dataset/yes/Deep/0584-2018_filtered-1baoguo-730648.jpg'

res=cv2.imread(image)



from PIL import Image
from torchvision import transforms
import torch

def apply_and_save_transforms(image_path, save_dir):
    # 读取图片
    image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
    
    # 定义transforms
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(15),
        transforms.RandomAdjustSharpness(2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # 应用transforms并保存结果
    for i, transform in enumerate(transform_list):
        image = transform(image)
        
        # 将Tensor转换回PIL Image以便保存
        if isinstance(image, torch.Tensor):
            image_to_save = transforms.ToPILImage()(image)
        else:
            image_to_save = image
        
        # 保存图片
        save_path = f"{save_dir}/transformed_image_{i}.png"
        image_to_save.save(save_path)
        print(f"Saved transformed image {i} to {save_path}")

# 示例调用
apply_and_save_transforms(image, './image_transform')


