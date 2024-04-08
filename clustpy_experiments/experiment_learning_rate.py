import torch
from clustpy.data import load_mnist, load_usps, load_imagenet10
from clustpy.deep import detect_device, DEC, DCN, IDEC, ACeDeC
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.utils import EvaluationAlgorithm, evaluate_multiple_datasets
from clustpy_experiments.utilities import _get_evaluation_datasets_with_autoencoders


def experiment_learning_rate(datasets, algorithms_function, metrics, save_dir, download_path,
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
        EvaluationAlgorithm("DEC_LR_1e-5", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DEC_LR_5e-5", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DEC_LR_5e-4", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DEC_LR_1e-3", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-3}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("IDEC_LR_1e-5", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("IDEC_LR_5e-5", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("IDEC_LR_5e-4", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("IDEC_LR_1e-3", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-3}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DCN_LR_1e-5", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DCN_LR_5e-5", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DCN_LR_5e-4", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("DCN_LR_1e-3", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-3}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("ACeDeC_LR_1e-5", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("ACeDeC_LR_5e-5", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 5e-5}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("ACeDeC_LR_1e-4", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
        EvaluationAlgorithm("ACeDeC_LR_1e-3", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-3}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size}),
    ]
    evaluate_multiple_datasets(evaluation_datasets, evaluation_algorithms, metrics, n_repetitions,
                               add_runtime=True, add_n_clusters=False,
                               save_path=save_dir + experiment_name + "/Results/result_lr.csv",
                               save_intermediate_results=True,
                               save_labels_path=save_dir + experiment_name + "/Labels/label_lr.csv")
