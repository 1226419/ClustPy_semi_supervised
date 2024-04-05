"""
@authors:
Lukas Miklautz
"""

import torch
import numpy as np
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc import ENRC
from typing import Union, Callable


class ACeDeC(ENRC):
    """
    Autoencoder Centroid-based Deep Cluster (ACeDeC) can be seen as a special case of ENRC where we have one
    cluster space and one shared space with a single cluster.

    Parameters
    ----------
    n_clusters : int
        number of clusters
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    P : list
        list containing projections for clusters in clustered space and cluster in shared space (optional) (default: None)
    input_centers : list
        list containing the cluster centers for clusters in clustered space and cluster in shared space (optional) (default: None)
    batch_size : int
        size of the data batches (default: 128)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder (default: 100)
    clustering_epochs : int
        maximum number of epochs for the actual clustering procedure (default: 150)
    tolerance_threshold : float
        tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
        for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
        will train as long as the labels are not changing anymore (default: None)
    optimizer_class : torch.optim.Optimizer
        optimizer for pretraining and training (default: torch.optim.Adam)
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction (default: torch.nn.MSELoss())
    degree_of_space_distortion : float
        weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure (default: 1.0)
    degree_of_space_preservation : float
        weight of regularization loss term, e.g., reconstruction loss (default: 1.0)
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new autoencoder will be created and trained (default: None)
    embedding_size : int
        size of the embedding within the autoencoder. Only used if autoencoder is None (default: 20)
    init : str
        choose which initialization strategy should be used. Has to be one of 'acedec', 'subkmeans', 'random' or 'sgd' (default: 'acedec')
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    device : torch.device
        if device is None then it will be checked whether a gpu is available or not (default: None)
    scheduler : torch.optim.lr_scheduler
        learning rate scheduler that should be used (default: None)
    scheduler_params : dict
        dictionary of the parameters of the scheduler object (default: None)
    init_kwargs : dict
        additional parameters that are used if init is a callable (optional) (default: None)
    init_subsample_size: int
        specify if only a subsample of size 'init_subsample_size' of the data should be used for the initialization. If None, all data will be used. (default: 10,000)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    final_reclustering : bool
        If True, the final embedding will be reclustered with the provided init strategy. (default: True)
    debug: bool
        if True additional information during the training will be printed (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    cluster_centers_ : np.ndarray
        The final cluster centers
    autoencoder : torch.nn.Module
        The final autoencoder

    Raises
    ----------
    ValueError : if init is not one of 'acedec', 'subkmeans', 'random', 'auto' or 'sgd'.

    References
    ----------
    Lukas Miklautz, Lena G. M. Bauer, Dominik Mautz, Sebastian Tschiatschek, Christian Böhm, Claudia Plant:
    Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces. IJCAI 2021: 2826-2832
    """

    def __init__(self, n_clusters: int, V: np.ndarray = None, P: list = None, input_centers: list = None,
                 batch_size: int = 128, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 tolerance_threshold: float = None, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                 degree_of_space_distortion: float = 1.0, degree_of_space_preservation: float = 1.0,
                 autoencoder: torch.nn.Module = None, embedding_size: int = 20, init: str = "acedec",
                 device: torch.device = None, scheduler: torch.optim.lr_scheduler = None,
                 scheduler_params: dict = None, init_kwargs: dict = None, init_subsample_size: int = 10000,
                 random_state: np.random.RandomState = None, custom_dataloaders: tuple = None, augmentation_invariance: bool = False,
                 final_reclustering: bool = True, debug: bool = False, fit_function: Union[Callable, str] = None,
                 clustering_module: torch.nn.Module = None):

        super().__init__([n_clusters, 1], V, P, input_centers,
                 batch_size, pretrain_optimizer_params, clustering_optimizer_params, pretrain_epochs, clustering_epochs,
                 tolerance_threshold, optimizer_class, loss_fn, degree_of_space_distortion, degree_of_space_preservation,
                 autoencoder, embedding_size, init, device, scheduler, scheduler_params, init_kwargs, init_subsample_size,
                 random_state, custom_dataloaders, augmentation_invariance, final_reclustering, debug, fit_function, clustering_module)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ACeDeC':
            """
            Cluster the input dataset with the ACeDeC algorithm. Saves the labels, centers, V, m, Betas, and P
            in the ACeDeC object.
            The resulting cluster labels will be stored in the labels_ attribute.
            Parameters
            ----------
            X : np.ndarray
                input data
            y : np.ndarray
                the labels (can be ignored)
            Returns
            ----------
            self : ACeDeC
                returns the AceDeC object
            """
            super().fit(X, y)
            self.labels_ = self.labels_[:, 0]
            self.acedec_labels_ = self.enrc_labels_[:, 0]
            return self

    def predict(self, X: np.ndarray, use_P: bool = True, dataloader: torch.utils.data.DataLoader = None) -> np.ndarray:
        """
        Predicts the labels of the input data.

        Parameters
        ----------
        X : np.ndarray
            input data
        use_P: bool
            if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used (default: True)
        dataloader : torch.utils.data.DataLoader
            dataloader to be used. Can be None if X is given (default: None)

        Returns
        -------
        predicted_labels : np.ndarray
            The predicted labels
        """
        predicted_labels = super().predict(X, use_P, dataloader)
        return predicted_labels[:, 0]
