import torch

from clustpy_experiments.utilities import _experiment
from clustpy.deep.autoencoders import ConvolutionalAutoencoder


def experiment_baseline(datasets, algorithms_function, metrics, save_dir, download_path,
                                      n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                                      n_clustering_epochs=50,
                                      optimizer_class=torch.optim.Adam,
                                      pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                                      other_ae_params={"conv_encoder_name": "resnet18"}):
    experiment_name = "BASELINE"
    embedding_size = 10
    ae_layers = [512]
    ae_class = ConvolutionalAutoencoder
    _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path)
