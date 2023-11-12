import numpy as np
from clustpy.utils import EvaluationDataset, EvaluationAlgorithm, EvaluationMetric
from clustpy.data import load_mnist, load_iris, load_banknotes, load_fmnist
from sklearn.model_selection import train_test_split
from clustpy.deep.autoencoders.convolutional_autoencoder import ConvolutionalAutoencoder
from clustpy.deep.autoencoders.feedforward_autoencoder import FeedforwardAutoencoder
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi

def get_dataloaders_from_config(config_dict: dict):
    percentages_of_unlabeled_data = config_dict["percentages"]
    perform_train_test_split = config_dict["train_test_split"]
    print(config_dict)
    list_of_datasets = []
    list_of_unique_datasets = []
    for dataset in config_dict["types"]:
        if dataset == "mnist":
            data, labels = load_mnist()
        elif dataset == "iris":
            data, labels = load_iris()
        elif dataset == "banknotes":
            data, labels = load_banknotes()
        elif dataset == "fmnist":
            data, labels = load_fmnist()
        else:
            raise ValueError("Dataset not found")
        if perform_train_test_split:
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
        else:
            X_train, X_test, y_train, y_test = data, labels, data, labels
        # add transformations here
        list_of_unique_datasets.append((X_train, X_test))
        for percentage in percentages_of_unlabeled_data:
            semi_supervised_labels = y_train.copy()
            semi_supervised_labels[np.random.choice(len(y_train), int(len(y_train) * percentage), replace=False)] = -1
            list_of_datasets.append(
                EvaluationDataset(f"{dataset}_" + str(int((1.0 - percentage) * 100)) + "_percent_labeled", X_train,
                                  labels_true=y_train, labels_train=semi_supervised_labels))
    return list_of_datasets, list_of_unique_datasets


def get_autoencoders_from_config(config_dict: dict, train_data):
    print(config_dict)
    print(train_data)

    list_of_trained_autoencoders = []
    for autoencoder in config_dict:
        autoencoder_type = list(autoencoder.keys())[0]
        optimizer_params = autoencoder[autoencoder_type]["optimizer_params"]
        layers = autoencoder[autoencoder_type]["layers"]
        n_epochs = autoencoder[autoencoder_type]["n_epochs"]
        input_height = autoencoder[autoencoder_type]["input_height"]
        if autoencoder_type == "convolutional":
            trained_autoencoder = ConvolutionalAutoencoder(input_height=input_height, fc_layers=layers,
                                                             reusable=True).fit(n_epochs=n_epochs,
                                                                                optimizer_params=optimizer_params,
                                                                                data=train_data)
        elif autoencoder_type == "feed_forward":
            trained_autoencoder = FeedforwardAutoencoder(layers=layers, reusable=True).fit(
                n_epochs=n_epochs, optimizer_params=optimizer_params, data=train_data)
        list_of_trained_autoencoders.append(trained_autoencoder)
    return list_of_trained_autoencoders



def get_algorithmns_from_config(config_dict: dict, list_of_autoencoders):
    print(config_dict)
def get_metrics_from_config(config_dict: dict):
    print(config_dict)
    list_of_metrics = []
    for metric in config_dict:
        if metric == "NMI":
            list_of_metrics.append(EvaluationMetric("NMI", nmi))
        elif metric == "ARI":
            list_of_metrics.append(EvaluationMetric("ARI", ari))
    return list_of_metrics
