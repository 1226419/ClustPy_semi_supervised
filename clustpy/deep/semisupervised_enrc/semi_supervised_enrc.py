"""
@authors:
Lukas Miklautz
"""

import torch
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
from clustpy.deep._utils import encode_batchwise, detect_device, set_torch_seed
from clustpy.deep._data_utils import get_dataloader, augmentation_invariance_check
from sklearn.utils import check_random_state
from clustpy.utils.plots import plot_scatter_matrix
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_init import available_init_strategies
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_fitting_procedure import apply_fitting_procedure, available_fitting_strategies
from clustpy.deep.semisupervised_enrc.helper_functions import enrc_predict_batchwise
from typing import Union, Callable

"""
===================== ENRC  =====================
"""


class ENRC(BaseEstimator, ClusterMixin):
    """
    The Embeddedn Non-Redundant Clustering (ENRC) algorithm.
        
    Parameters
    ----------
    n_clusters : list
        list containing number of clusters for each clustering
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    P : list
        list containing projections for each clustering (optional) (default: None)
    input_centers : list
        list containing the cluster centers for each clustering (optional) (default: None)
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
        choose which initialization strategy should be used. Has to be one of 'nrkmeans', 'random' or 'sgd' (default: 'nrkmeans')
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
        If True, the final embedding will be reclustered with the provided init strategy. (defaul: False)
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
    ValueError : if init is not one of 'nrkmeans', 'random', 'auto' or 'sgd'.

    References
    ----------
    Miklautz, Lukas & Dominik Mautz et al. "Deep embedded non-redundant clustering."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 04. 2020.
    """

    def __init__(self, n_clusters: list, V: np.ndarray = None, P: list = None, input_centers: list = None,
                 batch_size: int = 128, pretrain_optimizer_params: dict = None,
                 clustering_optimizer_params: dict = None, pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 tolerance_threshold: float = None, optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
                 degree_of_space_distortion: float = 1.0, degree_of_space_preservation: float = 1.0,
                 autoencoder: torch.nn.Module = None, embedding_size: int = 20, init: str = "nrkmeans",
                 device: torch.device = None, scheduler: torch.optim.lr_scheduler = None,
                 scheduler_params: dict = None, init_kwargs: dict = None, init_subsample_size: int = 10000,
                 random_state: np.random.RandomState = None, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, final_reclustering: bool = True, debug: bool = False,
                 fit_function: Union[Callable, str] = None, fit_kwargs: dict = None,
                 clustering_module: torch.nn.Module = None, reclustering_strategy: [Callable, str, None] = None):
        self.n_clusters = n_clusters.copy()
        self.device = device
        if self.device is None:
            self.device = detect_device()
        self.batch_size = batch_size
        self.pretrain_optimizer_params = {"lr":1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.clustering_optimizer_params = {"lr":1e-4} if clustering_optimizer_params is None else clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.tolerance_threshold = tolerance_threshold
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.degree_of_space_distortion = degree_of_space_distortion
        self.degree_of_space_preservation = degree_of_space_preservation
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.init_kwargs = init_kwargs
        self.init_subsample_size = init_subsample_size
        self.random_state = check_random_state(random_state)
        set_torch_seed(self.random_state)
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.final_reclustering = final_reclustering
        self.debug = debug
        self.fit_kwargs = fit_kwargs
        self.clustering_module = clustering_module
        if len(self.n_clusters) < 2:
            raise ValueError(f"n_clusters={n_clusters}, but should be <= 2.")
        if callable(init) or (init in available_init_strategies()):
            self.init = init
        else:
            raise ValueError(f"init={init} does not exist, has to be one of {available_init_strategies()}.")

        if callable(fit_function) or (fit_function in available_fitting_strategies()) or (fit_function is None):
            self.fit_function = fit_function
        else:
            raise ValueError(f"{fit_function} is not a function, is one of {available_fitting_strategies()} "
                             f"or is None")
        self.reclustering_strategy = reclustering_strategy
        self.input_centers = input_centers
        self.V = V
        self.m = None
        self.P = P

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ENRC':
        """
        Cluster the input dataset with the ENRC algorithm. Saves the labels, centers, V, m, Betas, and P
        in the ENRC object.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            input data
        y : np.ndarray
            the labels (can be ignored)
            
        Returns
        ----------
        self : ENRC
            returns the ENRC object
        """
        if (y is not None) and (self.init_kwargs is not None):
            self.init_kwargs["y"] = y
        elif (y is not None) and (self.init_kwargs is None):
            self.init_kwargs = {"y": y}

        augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)
        print("clustering_module", self.clustering_module)
        cluster_labels, cluster_centers, V, m, betas, P, n_clusters, autoencoder, cluster_labels_before_reclustering \
              = apply_fitting_procedure(X=X,
                                        n_clusters=self.n_clusters,
                                        V=self.V,
                                        P=self.P,
                                        input_centers=self.input_centers,
                                        batch_size=self.batch_size,
                                        pretrain_optimizer_params=self.pretrain_optimizer_params,
                                        clustering_optimizer_params=self.clustering_optimizer_params,
                                        pretrain_epochs=self.pretrain_epochs,
                                        clustering_epochs=self.clustering_epochs,
                                        tolerance_threshold=self.tolerance_threshold,
                                        optimizer_class=self.optimizer_class,
                                        loss_fn=self.loss_fn,
                                        degree_of_space_distortion=self.degree_of_space_distortion,
                                        degree_of_space_preservation=self.degree_of_space_preservation,
                                        autoencoder=self.autoencoder,
                                        embedding_size=self.embedding_size,
                                        init=self.init,
                                        random_state=self.random_state,
                                        device=self.device,
                                        scheduler=self.scheduler,
                                        scheduler_params=self.scheduler_params,
                                        init_kwargs=self.init_kwargs,
                                        init_subsample_size=self.init_subsample_size,
                                        custom_dataloaders=self.custom_dataloaders,
                                        augmentation_invariance=self.augmentation_invariance,
                                        final_reclustering=self.final_reclustering,
                                        debug=self.debug,
                                        fit_function=self.fit_function,
                                        fit_kwargs=self.fit_kwargs,
                                        clustering_module=self.clustering_module,
                                        reclustering_strategy=self.reclustering_strategy)

        # Update class variables
        self.labels_ = cluster_labels
        self.enrc_labels_ = cluster_labels_before_reclustering
        self.cluster_centers_ = cluster_centers
        self.V = V
        self.m = m
        self.P = P
        self.betas = betas
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        return self

    def predict(self, X: np.ndarray = None, use_P: bool = True, dataloader: torch.utils.data.DataLoader = None) -> np.ndarray:
        """
        Predicts the labels for each clustering of X in a mini-batch manner.
        
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
            n x c matrix, where n is the number of data points in X and c is the number of clusterings.
        """
        if dataloader is None:
            dataloader = get_dataloader(X, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.autoencoder.to(self.device)
        predicted_labels = enrc_predict_batchwise(V=torch.from_numpy(self.V).float().to(self.device),
                                                  centers=[torch.from_numpy(c).float().to(self.device) for c in
                                                           self.cluster_centers_],
                                                  subspace_betas=torch.from_numpy(self.betas).float().to(self.device),
                                                  model=self.autoencoder,
                                                  dataloader=dataloader,
                                                  device=self.device,
                                                  use_P=use_P)
        return predicted_labels

    def transform_full_space(self, X: np.ndarray, embedded=False) -> np.ndarray:
        """
        Embedds the input dataset with the autoencoder and the matrix V from the ENRC object.
        Parameters
        ----------
        X : np.ndarray
            input data
        embedded : bool
            if True, then X is assumed to be already embedded (default: False)
        
        Returns
        -------
        rotated : np.ndarray
            The transformed data
        """
        if not embedded:
            dataloader = get_dataloader(X, batch_size=self.batch_size, shuffle=False, drop_last=False)
            emb = encode_batchwise(dataloader=dataloader, module=self.autoencoder, device=self.device)
        else:
            emb = X
        rotated = np.matmul(emb, self.V)
        return rotated

    def transform_subspace(self, X: np.ndarray, subspace_index: int = 0, embedded: bool = False) -> np.ndarray:
        """
        Embedds the input dataset with the autoencoder and with the matrix V projected onto a special clusterspace_nr.
        
        Parameters
        ----------
        X : np.ndarray
            input data
        subspace_index: int
            index of the subspace_nr (default: 0)
        embedded: bool
            if True, then X is assumed to be already embedded (default: False)
        
        Returns
        -------
        subspace : np.ndarray
            The transformed subspace
        """
        if not embedded:
            dataloader = get_dataloader(X, batch_size=self.batch_size, shuffle=False, drop_last=False)
            emb = encode_batchwise(dataloader=dataloader, module=self.autoencoder, device=self.device)
        else:
            emb = X
        cluster_space_V = self.V[:, self.P[subspace_index]]
        subspace = np.matmul(emb, cluster_space_V)
        return subspace

    def plot_subspace(self, X: np.ndarray, subspace_index: int = 0, labels: np.ndarray = None, plot_centers: bool = False,
                      gt: np.ndarray = None, equal_axis: bool = False) -> None:
        """
        Plot the specified subspace_nr as scatter matrix plot.
       
        Parameters
        ----------
        X : np.ndarray
            input data
        subspace_index: int
            index of the subspace_nr (default: 0)
        labels: np.ndarray
            the labels to use for the plot (default: labels found by Nr-Kmeans) (default: None)
        plot_centers: bool
            plot centers if True (default: False)
        gt: np.ndarray
            of ground truth labels (default=None)
        equal_axis: bool
            equalize axis if True (default: False)
        Returns
        -------
        scatter matrix plot of the input data
        """
        if self.labels_ is None:
            raise Exception("The ENRC algorithm has not run yet. Use the fit() function first.")
        if labels is None:
            labels = self.labels_[:, subspace_index]
        if X.shape[0] != labels.shape[0]:
            raise Exception("Number of data objects must match the number of labels.")
        plot_scatter_matrix(self.transform_subspace(X, subspace_index), labels,
                            self.cluster_centers_[subspace_index] if plot_centers else None,
                            true_labels=gt, equal_axis=equal_axis)

    def reconstruct_subspace_centroids(self, subspace_index: int = 0) -> np.ndarray:
        """
        Reconstructs the centroids in the specified subspace_nr.

        Parameters
        ----------
        subspace_index: int
            index of the subspace_nr (default: 0)

        Returns
        -------
        centers_rec : centers_rec
            reconstructed centers as np.ndarray
        """
        cluster_space_centers = self.cluster_centers_[subspace_index]
        # rotate back as centers are in the V-rotated space
        centers_rot_back = np.matmul(cluster_space_centers, self.V.transpose())
        centers_rec = self.autoencoder.decode(torch.from_numpy(centers_rot_back).float().to(self.device))
        return centers_rec.detach().cpu().numpy()


