import torch
from clustpy.data import load_mnist, load_usps, load_imagenet10
from clustpy.deep import detect_device, DEC, DCN, IDEC
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.utils import EvaluationAlgorithm, evaluate_multiple_datasets
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from clustpy.partition import SubKmeans

from clustpy_experiments.utilities import _get_evaluation_datasets_with_autoencoders
from clustpy_experiments.additional_algos import AEEM, AESpectral, AESubKmeans


def experiment_initial_clustering(datasets, algorithms_function, metrics, save_dir, download_path,
                                  n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                  n_clustering_epochs=150,
                                  optimizer_class=torch.optim.Adam,
                                  pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                  other_ae_params={}):
    experiment_name = "FF"
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder

    ae_layers.append(embedding_size)
    experiment_name = experiment_name + "_" + "_".join(str(x) for x in ae_layers)
    dataset_loaders = [
        ("USPS", load_usps),
        ("MNIST", load_mnist),
        ("ImageNet10", load_imagenet10)
    ]
    device = detect_device()
    evaluation_datasets = _get_evaluation_datasets_with_autoencoders(dataset_loaders, ae_layers, experiment_name,
                                                                     n_repetitions, batch_size, n_pretrain_epochs,
                                                                     optimizer_class, pretrain_optimizer_params,
                                                                     loss_fn, ae_class, other_ae_params,
                                                                     device, save_dir, download_path)
    evaluation_algorithms = [
        EvaluationAlgorithm("AE+EM", AEEM,
                            {"n_clusters": None, "batch_size": batch_size}),
        EvaluationAlgorithm("DEC+EM", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": GaussianMixture,
                             "initial_clustering_params": {"n_init": 3}}),
        EvaluationAlgorithm("IDEC+EM", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": GaussianMixture,
                             "initial_clustering_params": {"n_init": 3}}),
        EvaluationAlgorithm("DCN+EM", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": GaussianMixture,
                             "initial_clustering_params": {"n_init": 3}}),
        EvaluationAlgorithm("AE+Spectral", AESpectral,
                            {"n_clusters": None, "batch_size": batch_size}),
        EvaluationAlgorithm("DEC+Spectral", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SpectralClustering}),
        EvaluationAlgorithm("IDEC+Spectral", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SpectralClustering}),
        EvaluationAlgorithm("DCN+Spectral", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SpectralClustering}),
        EvaluationAlgorithm("AE+SubKmeans", AESubKmeans,
                            {"n_clusters": None, "batch_size": batch_size}),
        EvaluationAlgorithm("DEC+SubKmeans", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SubKmeans}),
        EvaluationAlgorithm("IDEC+SubKmeans", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SubKmeans}),
        EvaluationAlgorithm("DCN+SubKmeans", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "initial_clustering_class": SubKmeans})
    ]
    evaluate_multiple_datasets(evaluation_datasets, evaluation_algorithms, metrics, n_repetitions,
                               add_runtime=True, add_n_clusters=False,
                               save_path=save_dir + experiment_name + "/Results/result_init_clust.csv",
                               save_intermediate_results=True,
                               save_labels_path=save_dir + experiment_name + "/Labels/label_init_clust.csv")