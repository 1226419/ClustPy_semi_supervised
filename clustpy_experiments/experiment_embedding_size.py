import torch
from clustpy.deep.autoencoders import FeedforwardAutoencoder

from clustpy_experiments.utilities import _experiment


def experiment_embedding_size(datasets, algorithms_function, metrics, save_dir, download_path,
                              n_repetitions=10, batch_size=256, n_pretrain_epochs=100,
                              n_clustering_epochs=150,
                              optimizer_class=torch.optim.Adam,
                              pretrain_optimizer_params={"lr": 1e-3}, loss_fn=torch.nn.MSELoss(),
                              other_ae_params={}):
    experiment_name = "EMBEDDING_SIZE"
    ae_layers = [500, 500, 2000]
    ae_class = FeedforwardAutoencoder
    for embedding_size in [2, 5, 25, 50, 100]:
        _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                    n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                    ae_class, other_ae_params, datasets, algorithms_function, metrics, save_dir, download_path)