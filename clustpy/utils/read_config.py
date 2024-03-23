import numpy as np
from clustpy.utils import EvaluationDataset, EvaluationAlgorithm, EvaluationMetric
from clustpy.data import load_mnist, load_iris, load_banknotes, load_fmnist
from sklearn.model_selection import train_test_split
from clustpy.deep.autoencoders.convolutional_autoencoder import ConvolutionalAutoencoder
from clustpy.deep.autoencoders.feedforward_autoencoder import FeedforwardAutoencoder
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from clustpy.deep.acedec import ACEDEC
import torch
import torchvision


def get_dataloaders_from_config(config_dict: dict):
    percentages_of_unlabeled_data = config_dict["percentages"]
    perform_train_test_split = config_dict["train_test_split"]
    print(config_dict)
    list_of_datasets = []
    list_of_unique_datasets = []
    for dataset in config_dict["list_of_datasets"]:
        dataset_name = list(dataset.keys())[0]
        print(dataset[dataset_name])
        if dataset_name == "mnist":
            tmp_dataset = load_mnist()
            data = tmp_dataset.images
            labels = tmp_dataset.target
            """
            data = data.reshape(-1, 1, 28, 28)
            data = torch.from_numpy(data).float()
            data = data.repeat(1, 3, 1, 1)
            padding_fn = torchvision.transforms.Pad([2, 2], fill=0)
            data = padding_fn(data)
            data /= 255.0
            data = data.numpy()
            """
        elif dataset_name == "iris":
            tmp_dataset = load_iris()
            data = tmp_dataset.images
            labels = tmp_dataset.target
        elif dataset_name == "banknotes":
            tmp_dataset = load_banknotes()
            data = tmp_dataset.images
            labels = tmp_dataset.target
        elif dataset_name == "fmnist":
            tmp_dataset = load_fmnist()
            data = tmp_dataset.images
            labels = tmp_dataset.target
        else:
            raise ValueError("Dataset not found")

        if "reshape" in list(dataset[dataset_name].keys()):
            reshape_params = dataset[dataset_name]["reshape"]
            data = data.reshape(reshape_params)
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
                EvaluationDataset(f"{dataset_name}_" + str(int((1.0 - percentage) * 100)) + "_percent_labeled", X_train,
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
        if autoencoder_type == "convolutional":
            input_height = autoencoder[autoencoder_type]["input_height"]
            channels = autoencoder[autoencoder_type]["channels"]
            trained_autoencoder = ConvolutionalAutoencoder(input_height=input_height, fc_layers=layers,
                                                             reusable=True, channels=channels).fit(n_epochs=n_epochs,
                                                                                optimizer_params=optimizer_params,
                                                                                data=train_data[0])
        elif autoencoder_type == "feed_forward":
            trained_autoencoder = FeedforwardAutoencoder(layers=layers, reusable=True).fit(
                n_epochs=n_epochs, optimizer_params=optimizer_params, data=train_data[0])
        list_of_trained_autoencoders.append(trained_autoencoder)
    return list_of_trained_autoencoders


def get_algorithms_from_config(config_dict: dict, list_of_autoencoders):
    print(config_dict)
    list_of_algorithms = []
    i = 0
    for autoencoder in list_of_autoencoders:
        for algorithms in config_dict:
            print(algorithms)
            algorithms_type = list(algorithms.keys())[0]
            if algorithms_type == "acedec":
                i += 1
                parameters = algorithms[algorithms_type]["params"]
                parameters["autoencoder"] = autoencoder
                print(parameters)
                list_of_algorithms.append(EvaluationAlgorithm("ACEDEC_"+str(i), ACEDEC, parameters))
    return list_of_algorithms


def get_metrics_from_config(config_dict: dict):
    print(config_dict)
    list_of_metrics = []
    for metric in config_dict:
        if metric == "NMI":
            list_of_metrics.append(EvaluationMetric("NMI", nmi))
        elif metric == "ARI":
            list_of_metrics.append(EvaluationMetric("ARI", ari))
    return list_of_metrics
