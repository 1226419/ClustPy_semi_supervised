from clustpy.deep.acedec import ACEDEC
from clustpy.data import load_mnist, load_iris, load_banknotes
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch
from clustpy.deep.autoencoders.flexible_autoencoder import FlexibleAutoencoder
import numpy as np
from clustpy.deep._data_utils import check_if_data_is_normalized
print("Hi World")
data, labels = load_banknotes()
print(labels)
#data, labels = load_mnist()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2) # random state can be added


def znorm(X):
    return (X - np.mean(X)) / np.std(X)

def minmax(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

check_if_data_is_normalized(X_train)
zdata = znorm(X_train)
check_if_data_is_normalized(zdata)
minmaxdata = minmax(X_train)
check_if_data_is_normalized(X_train)
data = znorm(X_train)

print(data.shape)
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(data)

my_ari = ari(y_train, kmeans.labels_)
print("ARI Training set Kmeans on raw data", my_ari)

my_nmi = nmi(y_train, kmeans.labels_)
print("NMI Training set Kmeans on raw data", my_nmi)
X_test_znorm = znorm(X_test)
labels_test = kmeans.predict(X_test_znorm)
my_ari = ari(labels_test, y_test)
print("ARI Test set Kmeans on raw data", my_ari)

my_nmi = nmi(labels_test, y_test)
print("NMI Test set Kmeans on raw data", my_nmi)

# setup smaller Autoencoder for faster training. Current default is [input_dim, 500, 500, 2000, embedding_size]
# small_autoencoder = None
# small_autoencoder = FlexibleAutoencoder(layers=[4, 2, 2, 2, 2]).fit(n_epochs=100, lr=1e-3, data=data)
small_autoencoder = FlexibleAutoencoder(layers=[4, 32, 8]).fit(n_epochs=100, lr=1e-3, data=data)
#medium_autoencoder = FlexibleAutoencoder(layers=[4, 126, 64, 32, 8]).fit(n_epochs=1000, lr=1e-3, data=data)
#small_autoencoder = FlexibleAutoencoder(layers=[784, 32, 20]).fit(n_epochs=100, lr=1e-3, data=data)

X_train_encoded = small_autoencoder.encode(torch.from_numpy(data).to(torch.float32)).detach().numpy()
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_train_encoded)

my_ari = ari(y_train, kmeans.labels_)
print("ARI Training set Kmeans on encoded training data", my_ari)

my_nmi = nmi(y_train, kmeans.labels_)
print("NMI Training set Kmeans on encoded training data", my_nmi)
X_test_encoded = small_autoencoder.encode(torch.from_numpy(X_test_znorm).to(torch.float32)).detach().numpy()
labels_test = kmeans.predict(X_test_encoded)
my_ari = ari(labels_test, y_test)
print("ARI Test set Kmeans on encoded training data", my_ari)

my_nmi = nmi(labels_test, y_test)
print("NMI Test set Kmeans on encoded training data", my_nmi)

dec = ACEDEC([2], autoencoder=small_autoencoder, debug=True, pretrain_epochs=2, clustering_epochs=1000, print_step=50)
# supervised fit
#dec.fit(data, labels)

# Create array of labels all having -1 vale
# unsupervised_labels = np.full(len(labels), -1)
# unsupervised fit
# dec.fit(data, unsupervised_labels)

# randomly assign some values of labels array to -1
semi_supervised_labels = y_train.copy()
# percentage 1.0 is unsupervised, 0.0 is supervised
percentage = 0.1
semi_supervised_labels[np.random.choice(len(y_train), int(len(y_train)*percentage), replace=False)] = -1

# semi-supervised fit
dec.fit(data, semi_supervised_labels)


#dec.fit(data)
print(dec.labels_)
predicted_labels = dec.labels_
y_train = y_train.astype(int)

difference = y_train - predicted_labels
print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0], (difference != 0).sum()))

my_ari = ari(y_train, dec.labels_)
print("ARI Training set", my_ari)

my_nmi = nmi(y_train, dec.labels_)
print("NMI Training set", my_nmi)

print("Test set")
labels_test = dec.predict(X_test_znorm)[:, 0]
print(labels_test)
print(y_test)

my_ari = ari(labels_test, y_test)
print("ARI Test set", my_ari)

my_nmi = nmi(labels_test, y_test)
print("NMI Test set", my_nmi)
