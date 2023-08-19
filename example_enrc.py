from clustpy.deep import ENRC
from clustpy.data import load_nrletters
from clustpy.data import load_iris
from clustpy.data import load_banknotes
from clustpy.data import load_mnist

from clustpy.utils import EvaluationDataset, EvaluationAlgorithm, EvaluationMetric, evaluate_multiple_datasets
from sklearn.metrics import adjusted_rand_score as ari
from clustpy.deep.autoencoders.flexible_autoencoder import FlexibleAutoencoder

"""
# Manual execution multi label
data, labels = load_nrletters()
enrc = ENRC([6, 4, 3], pretrain_epochs=10, clustering_epochs=10, verbose=True)
enrc.fit(data)
print(enrc.labels_)
print(labels)
difference = labels - enrc.labels_
print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0], (difference != 0).sum()))
"""
"""
# Automated execution
datasets = [EvaluationDataset("nr_Letters", data=load_nrletters)]
algorithms = [EvaluationAlgorithm("ENRC", ENRC, {"n_clusters": [6, 4, 3], "pretrain_epochs": 10, "clustering_epochs": 10})]
"""

"""
# Manual execution single label
data, labels = load_iris()
small_autoencoder = FlexibleAutoencoder(layers=[4, 4, 2]).fit(n_epochs=100, lr=1e-3, data=data)
dec = ENRC([3, 1], autoencoder=small_autoencoder, verbose=True)
dec.fit(data)
print(dec.labels_[:, 0])
print(labels)
my_ari = ari(labels, dec.labels_[:, 0])
print(my_ari)
"""

"""
data, labels = load_banknotes()

# setup smaller Autoencoder for faster training. Current default is [input_dim, 500, 500, 2000, embedding_size]
small_autoencoder = FlexibleAutoencoder(layers=[4, 32, 8]).fit(n_epochs=100, lr=1e-3, data=data)


dec = ENRC([2, 1], autoencoder=small_autoencoder, verbose=True, pretrain_epochs=1000, clustering_epochs=1000)
# supervised fit
dec.fit(data, labels)


#dec.fit(data)
predicted_labels = dec.labels_[:, 0]
labels = labels.astype(int)

difference = labels - predicted_labels
print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0], (difference != 0).sum()))

my_ari = ari(labels, dec.labels_[:, 0])
print(my_ari)
"""
data, labels = load_mnist()
print("data.shape", data.shape)
# setup smaller Autoencoder for faster training. Current default is [input_dim, 500, 500, 2000, embedding_size]
small_autoencoder = FlexibleAutoencoder(layers=[784, 32, 20]).fit(n_epochs=100, lr=1e-3, data=data)
enrc = ENRC([10, 1], pretrain_epochs=2, clustering_epochs=100, autoencoder=small_autoencoder,
debug=True)
# supervised fit
enrc.fit(data)
print(enrc.labels_[:, 0])
difference = labels - enrc.labels_[:, 0]
print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0], (difference != 0).sum()))