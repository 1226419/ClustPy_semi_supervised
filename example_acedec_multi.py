from clustpy.deep.acedec import ACEDEC
from clustpy.data import load_mnist, load_iris, load_banknotes
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from clustpy.metrics import unsupervised_clustering_accuracy as acc
from clustpy.deep.autoencoders.flexible_autoencoder import FlexibleAutoencoder
import numpy as np
from clustpy.utils import EvaluationDataset, EvaluationAlgorithm, EvaluationMetric, evaluate_multiple_datasets
from copy import deepcopy
print("Banknote multiple test runs")
data, labels = load_banknotes()

#data, labels = load_mnist()

percentages_of_unlabeled_data = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

datasets = []
data, labels = load_banknotes()
for percentage in percentages_of_unlabeled_data:
    semi_supervised_labels = labels.copy()
    semi_supervised_labels[np.random.choice(len(labels), int(len(labels) * percentage), replace=False)] = -1
    datasets.append(EvaluationDataset("Banknotes_"+str(int((1.0-percentage)*100))+ "_percent_labeled", data,
                                      labels_true=labels,
                                      y=semi_supervised_labels))


# setup smaller Autoencoder for faster training. Current default is [input_dim, 500, 500, 2000, embedding_size]
# datasets = [ EvaluationDataset("Banknotes", load_banknotes), EvaluationDataset("Iris", load_iris),
# EvaluationDataset("MNIST", load_mnist)]
small_autoencoder = FlexibleAutoencoder(layers=[4, 32, 8]).fit(n_epochs=100, lr=1e-3, data=data)
medium_autoencoder = FlexibleAutoencoder(layers=[4, 126, 64, 32, 8]).fit(n_epochs=100, lr=1e-3, data=data)
algorithmns = [
    EvaluationAlgorithm("ACEDEC_small_autoencoder", ACEDEC, {"n_clusters": [2], "autoencoder": deepcopy(
        small_autoencoder),
                         "debug":False, "clustering_epochs":10, "print_step": 50}),
    EvaluationAlgorithm("ACEDEC_small_autoencoder_100", ACEDEC, {"n_clusters": [2], "autoencoder": deepcopy(
        small_autoencoder),  "debug": False,  "clustering_epochs": 100,
                                                                  "print_step": 50}),
    EvaluationAlgorithm("ACEDEC_small_autoencoder_1000", ACEDEC, {"n_clusters": [2], "autoencoder": deepcopy(
        small_autoencoder),
                                                              "debug": False, "clustering_epochs": 1000,
                                                                   "print_step": 50}),
    EvaluationAlgorithm("ACEDEC_med_autoencoder_100", ACEDEC, {"n_clusters": [2], "autoencoder": deepcopy(
        medium_autoencoder),
                                                              "debug": False, "clustering_epochs": 100,
                                                                "print_step": 50}),
    EvaluationAlgorithm("ACEDEC_med_autoencoder_1000", ACEDEC, {"n_clusters": [2], "autoencoder": deepcopy(
        medium_autoencoder),
                                                              "debug": False,  "clustering_epochs": 1000,
                                                                 "print_step": 50})

]

#metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("ACC", acc),  EvaluationMetric("ARI", ari)]
metrics = [EvaluationMetric("NMI", nmi)]
#small_autoencoder = FlexibleAutoencoder(layers=[784, 32, 20]).fit(n_epochs=100, lr=1e-3, data=data)
#dec = ACEDEC([2], autoencoder=small_autoencoder, debug=True, pretrain_epochs=10, clustering_epochs=1000,
#             print_step=50)


df = evaluate_multiple_datasets(datasets, algorithmns, metrics, n_repetitions=5, aggregation_functions=[np.mean],
    add_runtime=False, add_n_clusters=False, save_path="evaluation_14_04_deepcopy_2.csv",
                                save_intermediate_results=False)
print(df)
