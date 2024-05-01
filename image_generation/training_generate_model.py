
import os
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset
import torch



# unets for unconditional imagen
# The code snippet you provided is defining a U-Net architecture using the `Unet` class from the
# `imagen_pytorch` library in Python. Here is a breakdown of the parameters used in the `Unet`
# constructor:
unet = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 1,
    layer_attns = (False, False, False, True),
    layer_cross_attns = False
)
# imagen, which contains the unet above
# This code snippet is creating an instance of the `Imagen` class, which is used for training an
# Imagen model. Here is a breakdown of the parameters passed to the `Imagen` constructor:
imagen = Imagen(
    condition_on_text = False,  # this must be set to False for unconditional Imagen
    unets = unet,
    image_sizes = 512,
    timesteps = 1000
)


# torch.cuda.set_device(1)
# torch.cuda.current_device()


trainer = ImagenTrainer(
    imagen = imagen,
    split_valid_from_train = True # whether to split the validation dataset from the training
).cuda()


# Imagen 模型的训练图像。
dataset = Dataset('/disk2/jialiangfan/dataset/medical_data/images', image_size = 512)
# print(f'len of dataset: {len(dataset)}')
trainer.add_train_dataset(dataset, batch_size = 4)
for i in range(200000):
    loss = trainer.train_step(unet_number = 1, max_batch_size = 4)
    print(f'loss: {loss}')
    if not (i % 50):
        valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
        print(f'valid loss: {valid_loss}')

    if not (i % 100) and trainer.is_main: # is_main makes sure this can run in distributed
        images = trainer.sample(batch_size = 1, return_pil_images = True) # returns List[Image]
        print(type(images),type(images[0]))
        images[0].save(f'./sample_images/sample-{i // 100}.png')
# save the trained model to disk
trainer.save('./head_imagen.pt')

    # # load the save model
    # model_dict= torch.load('./imagen.pt')
    # imagen.load_state_dict(model_dict)
    # # Generate an image using the model
    # images = imagen.sample(texts=[], cond_scale=3.)


