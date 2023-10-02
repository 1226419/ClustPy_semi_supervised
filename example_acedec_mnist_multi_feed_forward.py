from clustpy.deep.acedec import ACEDEC
from clustpy.data import load_mnist, load_iris, load_banknotes, load_fmnist
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from clustpy.deep.autoencoders.convolutional_autoencoder import ConvolutionalAutoencoder
from clustpy.deep.autoencoders.feedforward_autoencoder import FeedforwardAutoencoder
from clustpy.deep._data_utils import check_if_data_is_normalized
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.cluster import KMeans
from clustpy.utils import EvaluationDataset, EvaluationAlgorithm, EvaluationMetric, evaluate_multiple_datasets
print("Hi World")
import datetime
# Get the current date and time
current_datetime = datetime.datetime.now()
current_date = current_datetime.strftime('%Y_%m_%d')  # Format as YYYY-MM-DD
current_hour = current_datetime.hour
current_minute = current_datetime.minute
def znorm(X):
    return (X - np.mean(X)) / np.std(X)

def minmax(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

data, labels = load_mnist()
print(data.shape)
data = data.reshape(-1, 1, 28, 28)
print(data.shape)
print(len(labels))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2) # random state can be added
# setup convolutional Autoencoder for mnist training. Current default is [input_dim, 500, 500, 2000, embedding_size]
input_height = 28

if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device = torch.device('cpu')

X_train_minmax = minmax(X_train)
check_if_data_is_normalized(X_train_minmax)
X_train_minmax_kmeans_shape = X_train_minmax.reshape(len(X_train_minmax), 784)

#percentages_of_unlabeled_data = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
percentages_of_unlabeled_data = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
# percentages_of_unlabeled_data = [0.0, 1.0]
datasets = []
for percentage in percentages_of_unlabeled_data:
    semi_supervised_labels = y_train.copy()
    semi_supervised_labels[np.random.choice(len(y_train), int(len(y_train) * percentage), replace=False)] = -1
    datasets.append(EvaluationDataset("MNIST_"+str(int((1.0-percentage)*100)) + "_percent_labeled", X_train_minmax,
                                      labels_true=y_train, labels_train=semi_supervised_labels))

optimizer_params = {"lr": 1e-3}
# FEED forward needs data to be in different shape
feed_forward_layers = [input_height*input_height, 500, 500, 2000, 20]
feed_forward_autoencoder = FeedforwardAutoencoder(layers=feed_forward_layers, reusable=True).fit(n_epochs=100, optimizer_params=optimizer_params, data=X_train_minmax)


algorithmns = [

    EvaluationAlgorithm("ACEDEC_convolutional_autoencoder_200", ACEDEC, {"n_clusters": [10,1], "autoencoder":
        feed_forward_autoencoder,  "debug": False, "pretrain_epochs": 30, "clustering_epochs": 200,
                                                                   "print_step": 50, "recluster": True}),
    EvaluationAlgorithm("ACEDEC_convolutional_autoencoder_100", ACEDEC, {"n_clusters": [10,1], "autoencoder":
        feed_forward_autoencoder,  "debug": False, "pretrain_epochs": 30,  "clustering_epochs": 100,
                                                                 "print_step": 50, "recluster": True})

]
metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("ARI", ari)]
df = evaluate_multiple_datasets(datasets, algorithmns, metrics, n_repetitions=5, aggregation_functions=[np.mean],
    add_runtime=True, add_n_clusters=False, save_path=f"{current_date}_{current_hour}_{current_minute}_multi_mnist.csv",
                                save_intermediate_results=True)
print(df)