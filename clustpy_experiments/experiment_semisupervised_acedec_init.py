
import torch
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy_experiments.utilities import _get_evaluation_datasets_with_autoencoders
from clustpy_experiments.utilities import _experiment
from clustpy.deep import detect_device
from clustpy.utils import EvaluationAlgorithm, evaluate_multiple_datasets


def experiment_semi_init_small_feedforward(datasets, algorithms_function, metrics, save_dir, download_path,
                                          n_repetitions=10, batch_size=256, n_pretrain_epochs=200,
                                          n_clustering_epochs=150,
                                          optimizer_class=torch.optim.Adam,
                                          pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                          other_ae_params={}):
    experiment_name = "small_ff"
    embedding_size = 10
    ae_layers = [512, 256, 128]
    ae_class = FeedforwardAutoencoder
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path, # augmentation=True,
                train_labels_percent=10, train_test_split=True)


def experiment_semi_init_small_feedforward_multiple_label_percent(datasets, algorithms_function, metrics, save_dir, download_path,
                                          n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                          n_clustering_epochs=150,
                                          optimizer_class=torch.optim.Adam,
                                          pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                          other_ae_params={}):
    experiment_base_name = "small_ff"  # decides which autoencoders to use
    embedding_size = 10
    ae_layers = [512, 256, 128]
    ae_class = FeedforwardAutoencoder
    labels_percent_list = [10]
    for labels in labels_percent_list:
        experiment_name = experiment_base_name # + "_" + str(labels)
        _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                    n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                    ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path,
                    train_labels_percent=labels, train_test_split=True)


def experiment_semi_init_small_feedforward_multiple_label_percent_ds(datasets, algorithms_function, metrics, save_dir,
                                                                  download_path,
                                                                  n_repetitions=10, batch_size=256,
                                                                  n_pretrain_epochs=100,
                                                                  n_clustering_epochs=150,
                                                                  optimizer_class=torch.optim.Adam,
                                                                  pretrain_optimizer_params={"lr": 1e-3},
                                                                  loss_fn=torch.nn.MSELoss(),
                                                                  other_ae_params={}):
    device = detect_device()
    experiment_name = "small_ff"  # decides which autoencoders to use
    embedding_size = 10
    ae_layers = [512, 256, 128]
    ae_class = FeedforwardAutoencoder
    labels_percent_list = [0, 1, 5, 10, 50, 100]
    evaluation_datasets = []
    for labels in labels_percent_list:
        evaluation_datasets_tmp = _get_evaluation_datasets_with_autoencoders(datasets, ae_layers, experiment_name,
                                                                         n_repetitions, batch_size, n_pretrain_epochs,
                                                                         optimizer_class, pretrain_optimizer_params,
                                                                         loss_fn, ae_class, other_ae_params,
                                                                         device, save_dir, download_path,
                                                                             train_labels_percent=labels, train_test_split=True)
        evaluation_datasets.extend(evaluation_datasets_tmp)

    evaluate_multiple_datasets(evaluation_datasets, algorithms_function, metrics, n_repetitions,
                               add_runtime=True, add_n_clusters=False,
                               save_path=save_dir + experiment_name + "/Results/result_label_amount.csv",
                               save_intermediate_results=True,
                               save_labels_path=save_dir + experiment_name + "/Labels/label_label_amount.csv")


def experiment_semi_init_big_feedforward_multiple_label_percent(datasets, algorithms_function, metrics, save_dir, download_path,
                                          n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                          n_clustering_epochs=150,
                                          optimizer_class=torch.optim.Adam,
                                          pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                          other_ae_params={}):
    experiment_base_name = "big_ff"  # decides which autoencoders to use
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder
    labels_percent_list = [1, 5, 10, 15, 20, 50, 100]
    for labels in labels_percent_list:
        experiment_name = experiment_base_name  # + "_" + str(labels)
        _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                    n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                    ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path,
                    train_labels_percent=labels, train_test_split=True)


def experiment_semi_init_small_ff_multiple_label_percent_aug(datasets, algorithms_function, metrics, save_dir, download_path,
                                          n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                          n_clustering_epochs=150,
                                          optimizer_class=torch.optim.Adam,
                                          pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                          other_ae_params={}):
    experiment_base_name = "small_ff_aug"  # decides which autoencoders to use
    embedding_size = 10
    ae_layers = [512, 256, 128]
    ae_class = FeedforwardAutoencoder
    labels_percent_list = [10]
    augmentation = True

    experiment_name = experiment_base_name  # + "_" + str(labels)
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path,
                train_labels_percent=labels_percent_list, train_test_split=True, augmentation=augmentation)
