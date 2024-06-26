from clustpy.deep.enrc import ACeDeC
from clustpy.data import load_mnist, load_iris, load_banknotes, load_fmnist
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from clustpy.deep.autoencoders.convolutional_autoencoder import ConvolutionalAutoencoder
from clustpy.deep._data_utils import check_if_data_is_normalized
from sklearn.model_selection import train_test_split
from clustpy.deep import encode_batchwise, get_dataloader
import torchvision
import numpy as np
import torch
from sklearn.cluster import KMeans
print("Hi World")
import pytorch_warmup as warmup
from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineSchedulerWithLinearWarmup(object):
    def __init__(self, optimizer, warmup_period, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_period = warmup_period
        self.warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
    def step(self):
        with self.warmup_scheduler.dampening():
            self.cosine_scheduler.step()

def znorm(X):
    return (X - torch.mean(X)) / torch.std(X)


def minmax(X):
    return (X - torch.min(X)) / (torch.max(X) - torch.min(X))


def denormalize_fn(array: np.array, mean: float, std: float, w: int, h: int, c=1)->torch.Tensor:
    """
    This applies an inverse z-transformation and reshaping to visualize the images properly.
    """
    tensor = torch.from_numpy(array).float()
    pt_std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    pt_mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    return (tensor.mul(pt_std).add(pt_mean).view(-1, c, h, w) * 255).int().detach()

data, labels = load_mnist()
print(data.shape)
data = data.reshape(70000, 1, 28, 28)
print(data.shape)
print(len(labels))


# setup convolutional Autoencoder for mnist training. Current default is [input_dim, 500, 500, 2000, embedding_size]
input_height = 32
#fc_layers = [512, 500, 500, 2000, 20]
fc_layers = [512, 10]
#fc_layers = [512, 500, 500, 2000, 10]
if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device = torch.device('cpu')
data = torch.from_numpy(data).float()
data = data.repeat(1, 3, 1, 1)
padding_fn = torchvision.transforms.Pad([2, 2], fill=0)
data = padding_fn(data)
data /= 255.0

mean = data.mean()
std = data.std()
denormalize = lambda x: denormalize_fn(x, mean=mean, std=std, w=32, h=32, c=3)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2) # random state can be added

# preprocessing functions
normalize_fn = torchvision.transforms.Normalize([mean], [std])
flatten_fn = torchvision.transforms.Lambda(torch.flatten)
X_train_minmax = znorm(X_train)
check_if_data_is_normalized(X_train_minmax)
X_train_minmax_kmeans_shape = X_train_minmax.reshape(len(X_train_minmax), 3, input_height*input_height)
X_train_minmax_kmeans_shape = X_train_minmax_kmeans_shape[:, 0, :].squeeze()
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X_train_minmax_kmeans_shape)

# preprocessing functions
normalize_fn = torchvision.transforms.Normalize([mean], [std])
orig_transforms = torchvision.transforms.Compose([normalize_fn])

train_dl = get_dataloader(data, batch_size=256, shuffle=True,
                        ds_kwargs={"orig_transforms_list":[orig_transforms]},
                        )
# setup Augmentation dataloader
transform_list = [
    torchvision.transforms.ToPILImage(),
    # Perform it before cropping causes more computational overhead, but produces less artifacts
    torchvision.transforms.RandomAffine(degrees=(-16, +16),
                                                translate=(0.1, 0.1),
                                                shear=(-8, 8),
                                                fill=0),
    torchvision.transforms.ToTensor(),
    normalize_fn,
]

aug_transforms = torchvision.transforms.Compose(transform_list)
aug_train_dl = get_dataloader(data, batch_size=256, shuffle=True,
                        ds_kwargs={"aug_transforms_list": [aug_transforms], "orig_transforms_list": [orig_transforms]},
                        )


dl = get_dataloader(data, 256, shuffle=False,
                   ds_kwargs={"orig_transforms_list": [orig_transforms]})

my_ari = ari(y_train, kmeans.labels_)
print("ARI Training set Kmeans on raw data", my_ari)

my_nmi = nmi(y_train, kmeans.labels_)
print("NMI Training set Kmeans on raw data", my_nmi)
X_test_minmax = znorm(X_test)
X_test_minmax_kmeans = X_test_minmax.reshape(len(X_test_minmax), 3, input_height*input_height)
X_test_minmax_kmeans = X_test_minmax_kmeans[:, 0, :].squeeze()
labels_test = kmeans.predict(X_test_minmax_kmeans)
my_ari = ari(labels_test, y_test)
print("ARI Test set Kmeans on raw data", my_ari)

my_nmi = nmi(labels_test, y_test)
print("NMI Test set Kmeans on raw data", my_nmi)

print(device)
optimizer_params = {"lr": 1e-3}
weight_decay = 0.01
n_epochs = 50
warmup_factor = 0.3
warmup_period = int(warmup_factor*n_epochs)
scheduler = CosineSchedulerWithLinearWarmup
scheduler_params = {"warmup_period": warmup_period, "T_max": n_epochs, "verbose":True}
conv_autoencoder = ConvolutionalAutoencoder(input_height=input_height, fc_layers=fc_layers).fit(n_epochs=n_epochs,
                                                                                                optimizer_params=optimizer_params, dataloader=train_dl,
                                                                                                device=device, print_step=5,
           optimizer_class=lambda params, lr: torch.optim.AdamW(params, lr, weight_decay=weight_decay), scheduler=scheduler, scheduler_params=scheduler_params)

conv_autoencoder = conv_autoencoder.eval()# batch norm goes to another mode
# TODO: https://stackoverflow.com/questions/58447885/pytorch-going-back-and-forth-between-eval-and-train-modes
# https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
"""
X_train_encoded = conv_autoencoder.encode(torch.from_numpy(X_train_minmax).to(torch.float32).to(device)).detach().cpu().numpy()
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X_train_encoded)

my_ari = ari(y_train, kmeans.labels_)
print("ARI Training set Kmeans on encoded training data", my_ari)

my_nmi = nmi(y_train, kmeans.labels_)
print("NMI Training set Kmeans on encoded training data", my_nmi)
X_test_encoded = conv_autoencoder.encode(torch.from_numpy(X_test_minmax).to(torch.float32).to(device)).detach().cpu().numpy()
labels_test = kmeans.predict(X_test_encoded)
my_ari = ari(labels_test, y_test)
print("ARI Test set Kmeans on encoded training data", my_ari)

my_nmi = nmi(labels_test, y_test)
print("NMI Test set Kmeans on encoded training data", my_nmi)
"""
print("Convolutional Autoencoder created")
clustering_epochs = 100
warmup_factor = 0.3
warmup_period = int(warmup_factor*clustering_epochs)
# scheduler = CosineSchedulerWithLinearWarmup
# scheduler_params = {"warmup_period":warmup_period, "T_max":clustering_epochs, "verbose":False}
scheduler = torch.optim.lr_scheduler.StepLR
scheduler_params = {"step_size": int(0.2*clustering_epochs), "gamma": 0.5, "verbose": True}

dec = ACeDeC(10, autoencoder=conv_autoencoder, debug=True, pretrain_epochs=100, clustering_epochs=clustering_epochs, custom_dataloaders=[train_dl, dl],
             device=device, final_reclustering=True,          scheduler=scheduler, clustering_optimizer_params={'lr': 1e-3*0.5}, init_subsample_size=10000,
          scheduler_params=scheduler_params)
# supervised fit
#dec.fit(data, labels)

# Create array of labels all having -1 vale
# unsupervised_labels = np.full(len(labels), -1)
# unsupervised fit
# dec.fit(data, unsupervised_labels)

# randomly assign some values of labels array to -1
semi_supervised_labels = y_train.copy()
# percentage 1.0 is unsupervised, 0.0 is supervised
percentage = 0.0
semi_supervised_labels[np.random.choice(len(y_train), int(len(y_train)*percentage), replace=False)] = -1
print("acedec created")
dec.fit(data)
acedec_ari = ari(labels, dec.labels_)
print(f"ARI: {acedec_ari}")
print(f"ARI-no-reclustering: {ari(labels, dec.acedec_labels_)}")
"""
# semi-supervised fit
dec.fit(X_train, semi_supervised_labels)
print("acedec fit")

#dec.fit(data)
predicted_labels = dec.labels_
print("checking training set")
y_train = y_train.astype(int)

difference = y_train - predicted_labels
print("Number of mislabeled points out of a total %d points : %d" % (difference.shape[0], (difference != 0).sum()))

my_ari = ari(y_train, predicted_labels)
print("ari", my_ari)

my_nmi = nmi(y_train, predicted_labels)
print("nmi", my_nmi)

print("checking test set")
labels_test = dec.predict(X_test_minmax)
print("labels_test", labels_test)

y_test = y_test.astype(int)

difference = y_test - labels_test
print("Number of mislabeled points out of a total %d points : %d" % (difference.shape[0], (difference != 0).sum()))

my_ari = ari(y_test, labels_test)
print("ari", my_ari)

my_nmi = nmi(y_test, labels_test)
print("nmi", my_nmi)

"""
