import torch
from clustpy.deep.autoencoders import FeedforwardAutoencoder

from clustpy_experiments.utilities import _experiment


def experiment_feedforward_test_train(datasets, algorithms_function, metrics, save_dir, download_path,
                                      n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                      n_clustering_epochs=150,
                                      optimizer_class=torch.optim.Adam,
                                      pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                      other_ae_params={}):
    experiment_name = "TEST_TRAIN"
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path,
                train_test_split=True)


def experiment_feedforward_512_256_128_10(datasets, algorithms_function, metrics, save_dir, download_path,
                                          n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                          n_clustering_epochs=150,
                                          optimizer_class=torch.optim.Adam,
                                          pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                          other_ae_params={}):
    experiment_name = "FF"
    embedding_size = 10
    ae_layers = [512, 256, 128]
    ae_class = FeedforwardAutoencoder
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path)


def experiment_feedforward_500_500_2000_10(datasets, algorithms_function, metrics, save_dir, download_path,
                                           n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                           n_clustering_epochs=150,
                                           optimizer_class=torch.optim.Adam,
                                           pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                           other_ae_params={}):
    experiment_name = "FF"
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path)


def experiment_feedforward_500_500_2000_10_aug(datasets, algorithms_function, metrics, save_dir, download_path,
                                               n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                               n_clustering_epochs=150,
                                               optimizer_class=torch.optim.Adam,
                                               pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                               other_ae_params={}):
    experiment_name = "FF_AUG"
    embedding_size = 10
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder
    augmentation = True
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path,
                augmentation)
