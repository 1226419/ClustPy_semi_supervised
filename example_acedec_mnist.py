from cluspy.deep.acedec import ACEDEC
from cluspy.data import load_mnist, load_iris, load_banknotes, load_fmnist
from sklearn.metrics import adjusted_rand_score as ari
from cluspy.deep.convolutional_autoencoder import ConvolutionalAutoencoder
import numpy as np
import torch
print("Hi World")


data, labels = load_mnist()
print(data)
print(data.shape)
data = data.reshape(70000, 1, 28, 28)
print(data)
print(data.shape)
print(len(labels))

# setup convolutional Autoencoder for mnist training. Current default is [input_dim, 500, 500, 2000, embedding_size]
input_height = 28
fc_layers = [512, 500, 500, 2000, 100]
if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device = torch.device('cpu')

print(device)
conv_autoencoder = ConvolutionalAutoencoder(input_height=input_height, fc_layers=fc_layers).fit(n_epochs=100,
                                                                                                lr=1e-3, data=data,
                                                                                                device=device)

conv_autoencoder = conv_autoencoder.eval()# batch norm goes to another mode
# TODO: https://stackoverflow.com/questions/58447885/pytorch-going-back-and-forth-between-eval-and-train-modes
# https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146

print("Convolutional Autoencoder created")
dec = ACEDEC([10], autoencoder=conv_autoencoder, verbose=True, pretrain_epochs=2, clustering_epochs=1000, print_step=50, device=device)
# supervised fit
#dec.fit(data, labels)

# Create array of labels all having -1 vale
# unsupervised_labels = np.full(len(labels), -1)
# unsupervised fit
# dec.fit(data, unsupervised_labels)

# randomly assign some values of labels array to -1
semi_supervised_labels = labels.copy()
# percentage 1.0 is unsupervised, 0.0 is supervised
percentage = 0.0
semi_supervised_labels[np.random.choice(len(labels), int(len(labels)*percentage), replace=False)] = -1
print("acedec created")
# semi-supervised fit
dec.fit(data, semi_supervised_labels)
print("acedec fit")

#dec.fit(data)
predicted_labels = dec.labels_
labels = labels.astype(int)

difference = labels - predicted_labels
print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0], (difference != 0).sum()))

my_ari = ari(labels, dec.labels_)
print(my_ari)