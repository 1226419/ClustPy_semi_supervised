from clustpy.deep.acedec import ACEDEC
from clustpy.data import load_mnist, load_iris, load_banknotes
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from clustpy.deep.autoencoders.flexible_autoencoder import FlexibleAutoencoder
import numpy as np
print("Hi World")
data, labels = load_banknotes()
print(labels)
#data, labels = load_mnist()

def znorm(X):
    return (X - np.mean(X)) / np.std(X)
data = znorm(data)


# setup smaller Autoencoder for faster training. Current default is [input_dim, 500, 500, 2000, embedding_size]
# small_autoencoder = None
# small_autoencoder = FlexibleAutoencoder(layers=[4, 2, 2, 2, 2]).fit(n_epochs=100, lr=1e-3, data=data)
small_autoencoder = FlexibleAutoencoder(layers=[4, 32, 8]).fit(n_epochs=100, lr=1e-3, data=data)
medium_autoencoder = FlexibleAutoencoder(layers=[4, 126, 64, 32, 8]).fit(n_epochs=1000, lr=1e-3, data=data)
#small_autoencoder = FlexibleAutoencoder(layers=[784, 32, 20]).fit(n_epochs=100, lr=1e-3, data=data)


dec = ACEDEC([2], autoencoder=medium_autoencoder, debug=True, pretrain_epochs=2, clustering_epochs=1000, print_step=50)
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

# semi-supervised fit
dec.fit(data, semi_supervised_labels)


#dec.fit(data)
print(dec.labels_)
predicted_labels = dec.labels_
labels = labels.astype(int)

difference = labels - predicted_labels
print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0], (difference != 0).sum()))

my_ari = ari(labels, dec.labels_)
print(my_ari)

my_nmi = nmi(labels, dec.labels_)
print(my_nmi)
