import argparse


model_types = {
    0: "Vision Transformer",
    1: "ResNet18",
    2: "Vgg16",
    3: "Alexnet",
    4: "SqueezeNet",
    5: "ResNet34",
    6: "ResNet50",
    7: "DenseNet",
    8: "ViT_Base patch 16",
    9: "ViT_Base patch 32",
    10: "ViT_Large patch 16",
    11: "ViT_Large patch 32",
    12: "ViT_Huge patch 14",
    13: "ViT_base_MLP",
    14: "ViT_large_MLP",
    15: "ViT Adapter"
}

optimizer_types = {
    0: "SGD",
    1: "Adam",
    2: "AdamW"
}

device_types = {
    0: "cuda:0",
    1: "cuda:1"
}


model_help = 'Model to use for training: ' + ', '.join([f'{k}: {v}' for k, v in model_types.items()])
optimizer_help = 'Optimizer type: ' + ', '.join([f'{k}: {v}' for k, v in optimizer_types.items()])
device_help = 'Device to use: ' + ', '.join([f'{k}: {v}' for k, v in device_types.items()])


parser = argparse.ArgumentParser(description='Training Configuration')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--data_augmentation', type=bool, default=False, help='Use data augmentation or not')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model or not')
parser.add_argument('--optimizer_type', type=int, default=0, choices=optimizer_types.keys(), help=optimizer_help)
parser.add_argument('--device', type=int, default=0, choices=device_types.keys(), help=device_help)
parser.add_argument('--model', type=int, default=0, choices=model_types.keys(), help=model_help)

