import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def read_and_plot(filename,lr):
    with open(filename, "r") as file:
        text = file.read()
    # Manually splitting the input text by lines to better handle the extraction step-by-step

    lines = text.split("\n")

    # Lists to hold the extracted data
    training_losses = []
    training_accs = []
    validation_losses = []
    validation_accs = []

    # Iterating over lines and extracting the necessary parts
    for line in lines:
        # Extract training loss and accuracy
        train_match = re.search(r"Training Loss: ([\d.]+) Acc: ([\d.]+)%", line)
        if train_match:
            training_losses.append(float(train_match.group(1)))
            training_accs.append(float(train_match.group(2)))
        
        # Extract validation loss and accuracy
        val_match = re.search(r"Validation Loss: ([\d.]+) Acc: ([\d.]+)%", line)
        if val_match:
            validation_losses.append(float(val_match.group(1)))
            validation_accs.append(float(val_match.group(2)))

    # Creating a DataFrame with the extracted data
    df= pd.DataFrame({
        "Training Loss": training_losses,
        "Training Acc": training_accs,
        "Validation Loss": validation_losses,
        "Validation Acc": validation_accs
    })

    sns.set_theme(style="whitegrid")
    # 设置字体属性
    font_dict = {'family': 'Times New Roman', 'size': 15}
    plt.figure(figsize=(7, 5))
    font_dict={'family':'Times New Roman','size':15}



    # train_epochs = [int(epoch)+1 for epoch in train_epochs]
    # test_epochs = [int(epoch)+1 for epoch in test_epochs]

    sns.lineplot(df['Training Loss'],label="training loss",  linewidth=2.5, linestyle='-', marker='o', markersize=8, color='blue')  # 使用 seaborn 的 lineplot
    sns.lineplot(df['Validation Loss'],label="validating loss", linewidth=2.5,linestyle='--', marker='s', markersize=8, color='orange')  # 使用 seaborn 的 lineplot
    plt.title(title,fontdict=font_dict)

    plt.xlabel('Epoch',fontdict=font_dict)
    plt.ylabel('Loss',fontdict=font_dict)
    plt.legend(prop=font_dict)
    plt.tick_params(axis='both', labelsize=font_dict['size'])  # 设置坐标轴刻度标签的字体属性
    plt.savefig('loss-{}.png'.format(lr),dpi=400,bbox_inches='tight')
    plt.show()
    # Plotting the loss curves
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['Training Loss'], label='Training Loss', marker='o')
    # plt.plot(df['Validation Loss'], label='Validation Loss', marker='s')
    # plt.title('Training and Validation Loss Over Epochs (Learning Rate: 10^{-3})')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
# 'Training and Validation Loss Over Epochs (Learning Rate: 10^{-3})'
    # plt.savefig('loss-{}.png'.format(lr))
    # plt.show()

    # Plotting the accuracy curves
    plt.figure(figsize=(10, 6))

    plt.plot(df['Training Acc'], label='Training Accuracy', marker='o')
    plt.plot(df['Validation Acc'], label='Validation Accuracy', marker='s')
    plt.title('Training and Validation Accuracy Over Epochs (Learning Rate: {})'.format(lr))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('accuracy-{}.png'.format(lr))
    # plt.show()

if __name__ == "__main__":
    read_and_plot("vit-large-patch16-batch32-epoch10-lr0.001.o892229",lr=0.001)
    read_and_plot("vit-large-patch16-batch32-epoch10.o892228",lr=2e-05)
    read_and_plot("baseline.out",lr=22)
    read_and_plot("lr0.01",lr=0.01)
    read_and_plot("batch8",lr=0.008)

