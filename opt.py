import argparse

def read_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--data_argument', type=bool, default=False, help='using data arguemtnation or not')
    parser.add_argument('--model', type=int, default=0, help='Model to use for training: {0: ViT, 1: ResNet}')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained model or not')
    parser.add_argument('--optimizer_type', type=int, default=0, help='{0: SGD, 1: Adam}')

    # Add more arguments as needed

    args = parser.parse_args()
    return args

# main_args = read_args()
# print(main_args.data_argument)