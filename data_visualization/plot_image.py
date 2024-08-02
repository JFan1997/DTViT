import matplotlib.pyplot as plt
import torchvision


import matplotlib.pyplot as plt
import torchvision

class_names1 = ['no', 'yes']
class_names2 = ['Deep', 'Lobar','Subtentorial']

# plt.rcParams['figure.figsize'] = [12, 8]
# plt.rcParams['figure.dpi'] = 60
# plt.rcParams.update({'font.size': 20})
# def imshow(input, title):
#     # torch.Tensor => numpy
#     input = input.numpy().transpose((1, 2, 0))
#     # undo image normalization
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     input = std * input + mean
#     input = np.clip(input, 0, 1)
#     # display images
#     plt.imshow(input)
#     plt.title(title)
#     plt.show()
# ##Testing
# model.eval()
# model.to(device)
# start_time = time.time()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# with torch.no_grad():
#     running_loss = 0.
#     running_corrects = 0
#     for index, inputs  in enumerate(test_dataloader):
#         image, labels1,labels2 = inputs['pixel_values'],inputs['label1'],inputs['label2']
#         image = image.to(device)
#         labels1= labels1.to(device)
#         labels2= labels2.to(device)
#         outputs1,outputs2  = model(image)
#         _, preds1 = torch.max(outputs1, 1)
#         _, preds2 = torch.max(outputs2, 1)

#         loss1 = criterion1(outputs1, labels1)
#         loss2 = criterion2(outputs2, labels2)
#         loss = loss1 + loss2
#         running_loss += loss.item() * image.size(0)
#         running_corrects += torch.sum(preds1 == labels1.data)
#         running_corrects += torch.sum(preds2 == labels2.data)
#         if index == 0:
#             print('======>RESULTS<======')
#             images = torchvision.utils.make_grid(image[:4])
#             imshow(images.cpu(), title=[class_names1[x] for x in labels1[:4]])
#             imshow(images.cpu(), title=[class_names2[x] for x in labels2[:4]])
#     epoch_loss = running_loss / len(test_dataset)
#     epoch_acc = running_corrects / (2* len(test_dataset)) * 100.
#     print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.
#           format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

import numpy as np  
def plot_image(image,labels1,labels2,preds,preds2):
    class_names1 = {0:'no',1:'yes'}
    class_names2 = {0:'Deep',1:'Lobar',2:'Subtentorial',3:'no'}
    # The line `class_names2 = ['Deep', 'Lobar','Subtentorial']` is creating a list called
    # `class_names2` containing the class names 'Deep', 'Lobar', and 'Subtentorial'. This list is used
    # to map the numerical labels to their corresponding class names for visualization purposes in the
    # `plot_image` function.
    #  ['no', 'yes']
    # class_names2 = ['Deep', 'Lobar','Subtentorial']
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams.update({'font.size': 20})
    images = torchvision.utils.make_grid(image[:4])
    # title1="filename: {} \n real label: {} \n predicted label: {}"\
    #     .format([f.split('/')[-1] for f in filename] ,[class_names1[x] for x in labels1[:4]],[class_names1[y] for y in preds[:4]])
    # imshow(images.cpu(), title1)
    # title2="filename: {} \n real label: {} \n predicted label: {}"\
    #     .format([f.split('/')[-1] for f in filename] ,[class_names2[x] for x in labels2[:4]],[class_names2[y] for y in preds2[:4]])
    # imshow(images.cpu(), title)
    title1="real label: {} \n predicted label: {}"\
        .format([class_names1[x] for x in labels1],[class_names1[y] for y in preds])
    imshow(images.cpu(), title1)
    title2="real label: {} \n predicted label: {}"\
        .format([class_names2[x] for x in labels2],[class_names2[y] for y in preds2])
    imshow(images.cpu(), title2)


    # imshow(images.cpu(), title=[class_names1[x] for x in labels1[:4]])
    # imshow(images.cpu(), title=[class_names2[x] for x in labels2[:4]])


def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()
