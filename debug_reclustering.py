# Importing all necessary libraries

# internal packages

import os
from collections import Counter
# external packages
import torch
import torchvision
import numpy as np
import sklearn
from sklearn.metrics import normalized_mutual_info_score
from matplotlib import pyplot as plt

import seaborn as sns
import pandas as pd
import clustpy
from clustpy.deep import encode_batchwise, get_dataloader
from clustpy.alternative import NrKmeans
from clustpy.metrics.multipe_labelings_scoring import MultipleLabelingsConfusionMatrix
import joblib

# specify base paths

base_path = "/mnt/data/miklautzl92dm_data/clustpy/material"
model_name = "acedec_convae.pth"

print("Versions")
print("torch: ", torch.__version__)
print("torchvision: ", torchvision.__version__)
print("numpy: ", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("clustpy:", clustpy.__version__)


# Some helper functions, you can ignore those in the beginning
def denormalize_fn(array: np.array, mean: float, std: float, w: int, h: int, c=1) -> torch.Tensor:
    """
    This applies an inverse z-transformation and reshaping to visualize the images properly.
    """
    tensor = torch.from_numpy(array).float()
    pt_std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    pt_mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    return (tensor.mul(pt_std).add(pt_mean).view(-1, c, h, w) * 255).int().detach()


def plot_images(images: torch.Tensor, pad: int = 1):
    """Aligns multiple images on an N by 8 grid"""

    def imshow(img):
        plt.figure(figsize=(10, 20))
        npimg = img.numpy()
        npimg = np.array(npimg)
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)),
                   vmin=0, vmax=1)

    imshow(torchvision.utils.make_grid(images, pad_value=255, normalize=False, padding=pad))
    plt.show()


def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    return device

from clustpy.data import load_mnist


# load data
dataset = load_mnist()
data = dataset.images
print(type(data))
print(data.shape)
print("1")

labels = dataset.target
data = data.reshape(-1, 1, 28, 28)
print(type(data))
print(data.shape)
print("2")

data = torch.from_numpy(data).float()
print(type(data))
print(data.shape)
print("3")


data = data.repeat(1,3,1,1)
print(type(data))
print(data.shape)
print("4")


padding_fn = torchvision.transforms.Pad([2,2], fill=0)
data = padding_fn(data)
print(type(data))
print(data.shape)
print("5")



data /= 255.0
print(type(data))
print(data.shape)
print("6")



mean = data.mean()
std = data.std()
denormalize = lambda x: denormalize_fn(x, mean=mean, std=std, w=32, h=32, c=3)


# preprocessing functions
normalize_fn = torchvision.transforms.Normalize([mean], [std])
flatten_fn = torchvision.transforms.Lambda(torch.flatten)

# augmentation transforms
transform_list = [
    torchvision.transforms.ToPILImage(),
    # Perform it before cropping causes more computational overhead, but produces less artifacts
    torchvision.transforms.RandomAffine(degrees=(-16,+16),
                                                translate=(0.1, 0.1),
                                                shear=(-8, 8),
                                                fill=0),
    torchvision.transforms.ToTensor(),
    normalize_fn,
]

aug_transforms = torchvision.transforms.Compose(transform_list)
orig_transforms = torchvision.transforms.Compose([normalize_fn])

# pass transforms to dataloader
aug_dl = get_dataloader(data, batch_size=32, shuffle=False,
                        ds_kwargs={"aug_transforms_list":[aug_transforms], "orig_transforms_list":[orig_transforms]},
                        )

from clustpy.deep.enrc import ACeDeC
from clustpy.deep.autoencoders import ConvolutionalAutoencoder
from sklearn.metrics import adjusted_rand_score as ari
device = torch.device("cuda:0")
ae = ConvolutionalAutoencoder(32, [512, 10], conv_encoder_name="resnet18").to(device)
model_path = "test-convae_reproduce_model_trained_100.pth"
sd = torch.load(model_path)
ae.load_state_dict(sd.state_dict())
ae.fitted = True
ae.to(device)

aug_train_dl = get_dataloader(data, batch_size=256, shuffle=True, additional_inputs=labels,
                        ds_kwargs={"aug_transforms_list":[aug_transforms, None], "orig_transforms_list":[orig_transforms, None]},
                        dl_kwargs={"num_workers":8})

train_dl = get_dataloader(data, batch_size=256, shuffle=True, additional_inputs=labels,
                        ds_kwargs={"orig_transforms_list":[orig_transforms, None]},
                        dl_kwargs={"num_workers":8})

dl = get_dataloader(data, 256, shuffle=False, additional_inputs=labels,
                   ds_kwargs={"orig_transforms_list":[orig_transforms, None]})

ae_lr = 1e-3
clustering_epochs = 5
warmup_factor = 0.3
warmup_period = int(warmup_factor*clustering_epochs)
# scheduler = CosineSchedulerWithLinearWarmup
# scheduler_params = {"warmup_period":warmup_period, "T_max":clustering_epochs, "verbose":False}
scheduler = torch.optim.lr_scheduler.StepLR
scheduler_params = {"step_size":int(0.2*clustering_epochs), "gamma":0.5, "verbose": True}



acedec = ACeDeC(n_clusters=10,
          clustering_epochs=clustering_epochs,
          autoencoder=ae,
          clustering_optimizer_params={'lr': ae_lr*0.5},
          custom_dataloaders=[train_dl, dl],
          scheduler=scheduler,
          scheduler_params=scheduler_params,
          final_reclustering=True,
          debug=True,
          init_subsample_size=10000,
          )

acedec.fit(data)

acedec_ari = ari(labels, acedec.labels_)
print(f"ARI: {acedec_ari}")
print(f"ARI-no-reclustering: {ari(labels, acedec.acedec_labels_)}")
rec_centers = acedec.reconstruct_subspace_centroids()
plot_images(denormalize(rec_centers))

# Soft Beta Weights
fig, ax = plt.subplots(figsize=(10,2))
sns.heatmap(acedec.betas, vmin=0, vmax=1.0, ax=ax)
ax.set_yticklabels(["$\\beta_0$", "$\\beta_1$"])
plt.show()

#######################################################################################

from clustpy.deep.acedec import ACEDEC
from clustpy.deep.autoencoders import ConvolutionalAutoencoder
from sklearn.metrics import adjusted_rand_score as ari
import torch
device = torch.device("cuda:0")
ae = ConvolutionalAutoencoder(32, [512, 10], conv_encoder_name="resnet18").to(device)
model_path = "test-convae_reproduce_model_trained_100.pth"
sd = torch.load(model_path)
ae.load_state_dict(sd.state_dict())
ae.fitted = True
ae.to(device)

aug_train_dl = get_dataloader(data, batch_size=256, shuffle=True, additional_inputs=labels,
                        ds_kwargs={"aug_transforms_list":[aug_transforms, None], "orig_transforms_list":[orig_transforms, None]},
                        dl_kwargs={"num_workers":8})

train_dl = get_dataloader(data, batch_size=256, shuffle=True, additional_inputs=labels,
                        ds_kwargs={"orig_transforms_list":[orig_transforms, None]},
                        dl_kwargs={"num_workers":8})

dl = get_dataloader(data, 256, shuffle=False, additional_inputs=labels,
                   ds_kwargs={"orig_transforms_list":[orig_transforms, None]})


ae_lr = 1e-3
clustering_epochs = 5
warmup_factor = 0.3
warmup_period = int(warmup_factor*clustering_epochs)
# scheduler = CosineSchedulerWithLinearWarmup
# scheduler_params = {"warmup_period":warmup_period, "T_max":clustering_epochs, "verbose":False}
scheduler = torch.optim.lr_scheduler.StepLR
scheduler_params = {"step_size":int(0.2*clustering_epochs), "gamma":0.5, "verbose": True}



my_acedec = ACEDEC(n_clusters=10,
          autoencoder=ae,
          clustering_optimizer_params={'lr': ae_lr*0.5, 'epochs': clustering_epochs},
          custom_dataloaders=[train_dl, dl],
          scheduler=scheduler,
          scheduler_params=scheduler_params,
          final_reclustering=True,
          debug=True,
          init_subsample_size=10000,
          reclustering_kwargs={'lr': ae_lr*0.5}
          )


my_acedec.fit(data)
my_acedec_ari = ari(labels, my_acedec.labels_)
print(f"ARI: {my_acedec_ari}")
print(f"ARI-no-reclustering: {ari(labels, my_acedec.acedec_labels_)}")

rec_centers = my_acedec.reconstruct_subspace_centroids()
plot_images(denormalize(rec_centers))

# Soft Beta Weights
fig, ax = plt.subplots(figsize=(10,2))
sns.heatmap(my_acedec.betas, vmin=0, vmax=1.0, ax=ax)
ax.set_yticklabels(["$\\beta_0$", "$\\beta_1$"])
plt.show()
