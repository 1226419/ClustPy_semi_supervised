from clustpy.deep.acedec import ACEDEC
from clustpy.data import load_mnist, load_iris, load_banknotes, load_fmnist
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from clustpy.deep.autoencoders.convolutional_autoencoder import ConvolutionalAutoencoder
from clustpy.deep._data_utils import check_if_data_is_normalized
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.cluster import KMeans
print("Hi World")
def znorm(X):
    return (X - np.mean(X)) / np.std(X)

def minmax(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

data, labels = load_mnist()
print(data)
print(data.shape)
data = data.reshape(70000, 1, 28, 28)
print(data)
print(data.shape)
print(len(labels))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2) # random state can be added
# setup convolutional Autoencoder for mnist training. Current default is [input_dim, 500, 500, 2000, embedding_size]
input_height = 28
fc_layers = [512, 500, 500, 2000, 10]
if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device = torch.device('cpu')

X_train_minmax = minmax(X_train)
check_if_data_is_normalized(X_train_minmax)
X_train_minmax_kmeans_shape = X_train_minmax.reshape(len(X_train_minmax), 784)
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X_train_minmax_kmeans_shape)

my_ari = ari(y_train, kmeans.labels_)
print("ARI Training set Kmeans on raw data", my_ari)

my_nmi = nmi(y_train, kmeans.labels_)
print("NMI Training set Kmeans on raw data", my_nmi)
X_test_znorm = znorm(X_test)
X_test_znorm_kmeans = X_test_znorm.reshape(len(X_test_znorm), 784)
labels_test = kmeans.predict(X_test_znorm_kmeans)
my_ari = ari(labels_test, y_test)
print("ARI Test set Kmeans on raw data", my_ari)

my_nmi = nmi(labels_test, y_test)
print("NMI Test set Kmeans on raw data", my_nmi)

print(device)
conv_autoencoder = ConvolutionalAutoencoder(input_height=input_height, fc_layers=fc_layers).fit(n_epochs=10,
                                                                                                lr=1e-3, data=X_train_minmax,
                                                                                                device=device)

conv_autoencoder = conv_autoencoder.eval()# batch norm goes to another mode
# TODO: https://stackoverflow.com/questions/58447885/pytorch-going-back-and-forth-between-eval-and-train-modes
# https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146

X_train_encoded = conv_autoencoder.encode(torch.from_numpy(X_train_minmax).to(torch.float32)).detach().numpy()
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X_train_encoded)

my_ari = ari(y_train, kmeans.labels_)
print("ARI Training set Kmeans on encoded training data", my_ari)

my_nmi = nmi(y_train, kmeans.labels_)
print("NMI Training set Kmeans on encoded training data", my_nmi)
X_test_encoded = conv_autoencoder.encode(torch.from_numpy(X_test_znorm).to(torch.float32)).detach().numpy()
labels_test = kmeans.predict(X_test_encoded)
my_ari = ari(labels_test, y_test)
print("ARI Test set Kmeans on encoded training data", my_ari)

my_nmi = nmi(labels_test, y_test)
print("NMI Test set Kmeans on encoded training data", my_nmi)


print("Convolutional Autoencoder created")
dec = ACEDEC([10], autoencoder=conv_autoencoder, debug=True, pretrain_epochs=2, clustering_epochs=10, print_step=50,
             device=device, recluster=True)
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
print("ari", my_ari)

my_nmi = nmi(labels, dec.labels_)
print("nmi", my_nmi)