import torch
import numpy as np
from clustpy.deep._utils import encode_batchwise, detect_device
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._train_utils import get_trained_autoencoder
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_init import apply_init_function
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_module import _ENRC_Module
from typing import Callable, Union


def enrc_fitting_with_labels(X: np.ndarray, n_clusters: list, V: np.ndarray, P: list, input_centers: list, batch_size: int,
              pretrain_optimizer_params: dict, clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
              tolerance_threshold: float,
              optimizer_class: torch.optim.Optimizer, loss_fn: torch.nn.modules.loss._Loss,
              degree_of_space_distortion: float, degree_of_space_preservation: float, autoencoder: torch.nn.Module,
              embedding_size: int, init: str, random_state: np.random.RandomState, device: torch.device,
              scheduler: torch.optim.lr_scheduler, scheduler_params: dict,  init_kwargs: dict,
              init_subsample_size: int, custom_dataloaders: tuple, augmentation_invariance: bool, final_reclustering: bool,
              debug: bool, clustering_module: torch.nn.Module,
                 reclustering_strategy: Union[Callable, str, None] = None, y:np.ndarray = None) -> (
        np.ndarray, list, np.ndarray, list, np.ndarray, list, list, torch.nn.Module):
    """
    Start the actual ENRC clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        input data
    n_clusters : list
        list containing number of clusters for each clustering
    V : np.ndarray
        orthogonal rotation matrix
    P : list
        list containing projections for each clustering
    input_centers : list
        list containing the cluster centers for each clustering
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    clustering_optimizer_params: dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder
    clustering_epochs : int
        maximum number of epochs for the actual clustering procedure
    optimizer_class : torch.optim.Optimizer
        optimizer for pretraining and training
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction
    degree_of_space_distortion : float
        weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure
    degree_of_space_preservation : float
        weight of regularization loss term, e.g., reconstruction loss
    autoencoder : torch.nn.Module
         the input autoencoder. If None a new autoencoder will be created and trained
    embedding_size : int
        size of the embedding within the autoencoder. Only used if autoencoder is None
    init : str
        strchoose which initialization strategy should be used. Has to be one of 'nrkmeans', 'random' or 'sgd'.
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution
    device : torch.device
        if device is None then it will be checked whether a gpu is available or not
    scheduler : torch.optim.lr_scheduler
        learning rate scheduler that should be used
    scheduler_params : dict
        dictionary of the parameters of the scheduler object
    tolerance_threshold : float
        tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
        for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
        will train as long as the labels are not changing anymore.
    init_kwargs : dict
        additional parameters that are used if init is a callable
    init_subsample_size : int
        specify if only a subsample of size 'init_subsample_size' of the data should be used for the initialization
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)
    final_reclustering : bool
        If True, the final embedding will be reclustered with the provided init strategy. (defaul: False)
    debug : bool
        if True additional information during the training will be printed

    Returns
    -------
    tuple : (np.ndarray, list, np.ndarray, list, np.ndarray, list, list, torch.nn.Module)
        the cluster labels,
        the cluster centers,
        the orthogonal rotation matrix,
        the dimensionalities of the subspaces,
        the betas,
        the projections of the subspaces,
        the final n_clusters,
        the final autoencoder
        the cluster labels before final_reclustering
    """
    # Set device to train on
    if device is None:
        device = detect_device()

    # Setup dataloaders
    if custom_dataloaders is None:
        if y is not None:
            trainloader = get_dataloader(X, batch_size, True, False, additional_inputs=y)
            testloader = get_dataloader(X, batch_size, False, False, additional_inputs=y)
        else:
            trainloader = get_dataloader(X, batch_size, True, False)
            testloader = get_dataloader(X, batch_size, False, False)
    else:
        trainloader, testloader = custom_dataloaders
        if debug: print("Custom dataloaders are used, X will be overwritten with testloader return values.")
        _preprocessed = []
        for batch in testloader: _preprocessed.append(batch[1])
        X = torch.cat(_preprocessed)

    if trainloader.batch_size != batch_size:
        if debug: print("WARNING: Specified batch_size differs from trainloader.batch_size. Will use trainloader.batch_size.")
        batch_size = trainloader.batch_size

    # Use subsample of the data if specified and subsample is smaller than dataset
    if init_subsample_size is not None and init_subsample_size > 0 and init_subsample_size < X.shape[0]:
        rand_idx = random_state.choice(X.shape[0], init_subsample_size, replace=False)
        subsampleloader = get_dataloader(X[rand_idx], batch_size=batch_size, shuffle=False, drop_last=False)
        if (init_kwargs is not None) and ("y" in init_kwargs.keys()):
            y_sampled_init_labels = init_kwargs["y"][rand_idx]
            init_kwargs["y"] = y_sampled_init_labels
    else:
        subsampleloader = testloader
    if debug: print("Setup autoencoder")
    # Setup autoencoder
    autoencoder = get_trained_autoencoder(trainloader, pretrain_optimizer_params, pretrain_epochs, device,
                                          optimizer_class, loss_fn, embedding_size, autoencoder)
    # Run ENRC init
    if debug:
        print("Run init: ", init)
        print("Start encoding")
    embedded_data = encode_batchwise(subsampleloader, autoencoder, device)
    if debug:
        print("Start initializing parameters")
    # set init epochs proportional to clustering_epochs
    init_epochs = np.max([10, int(0.2*clustering_epochs)])
    input_centers, P, V, beta_weights = apply_init_function(data=embedded_data, n_clusters=n_clusters, device=device,
                                                            init=init, rounds=10,
                                                            epochs=init_epochs, batch_size=batch_size, debug=debug,
                                                            input_centers=input_centers, P=P, V=V,
                                                            random_state=random_state,
                                                            max_iter=100, optimizer_params=clustering_optimizer_params,
                                                            optimizer_class=optimizer_class, init_kwargs=init_kwargs,
                                                            )
    if int(sum(y)) == len(y) * -1:
        print("no labels found in clustering module falling back to enrc clustering module")
        clustering_module = _ENRC_Module
        y = None
    # Setup ENRC Module
    enrc_module = clustering_module(input_centers, P, V, degree_of_space_distortion=degree_of_space_distortion,
                               degree_of_space_preservation=degree_of_space_preservation,
                               beta_weights=beta_weights, augmentation_invariance=augmentation_invariance).to_device(device)
    if debug:
        print("Betas after init")
        print(enrc_module.subspace_betas().detach().cpu().numpy())
    # In accordance to the original paper we update the betas 10 times faster
    clustering_optimizer_beta_params = clustering_optimizer_params.copy()
    clustering_optimizer_beta_params["lr"] = clustering_optimizer_beta_params["lr"] * 10
    param_dict = [dict({'params': autoencoder.parameters()}, **clustering_optimizer_params),
                  dict({'params': [enrc_module.V]}, **clustering_optimizer_params),
                  dict({'params': [enrc_module.beta_weights]}, **clustering_optimizer_beta_params)
                  ]
    optimizer = optimizer_class(param_dict)

    if scheduler is not None:
        scheduler = scheduler(optimizer, **scheduler_params)

    # Training loop
    if debug:
        print("Start training")
    if y is None:
        enrc_module.fit(trainloader=trainloader,
                        evalloader=testloader,
                        max_epochs=clustering_epochs,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        batch_size=batch_size,
                        model=autoencoder,
                        device=device,
                        scheduler=scheduler,
                        tolerance_threshold=tolerance_threshold,
                        debug=debug)
    else:
        enrc_module.fit(trainloader=trainloader,
                        evalloader=testloader,
                        max_epochs=clustering_epochs,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        batch_size=batch_size,
                        model=autoencoder,
                        device=device,
                        scheduler=scheduler,
                        tolerance_threshold=tolerance_threshold,
                        debug=debug, y=y)

    if debug:
        print("Betas after training")
        print(enrc_module.subspace_betas().detach().cpu().numpy())

    cluster_labels_before_reclustering = enrc_module.predict_batchwise(model=autoencoder, dataloader=testloader, device=device, use_P=True)
    # Recluster
    if final_reclustering:
        if debug:
            print("Recluster")
        if reclustering_strategy is None:
            reclustering_strategy = init
        enrc_module.recluster(dataloader=subsampleloader, model=autoencoder, device=device, optimizer_params=clustering_optimizer_params,
                              optimizer_class=optimizer_class, reclustering_strategy=reclustering_strategy, init_kwargs=init_kwargs)
        # Predict labels and transfer other parameters to numpy
        cluster_labels = enrc_module.predict_batchwise(model=autoencoder, dataloader=testloader, device=device, use_P=True)
        if debug:
            print("Betas after reclustering")
            print(enrc_module.subspace_betas().detach().cpu().numpy())
    else:
        cluster_labels = cluster_labels_before_reclustering
    cluster_centers = [centers_i.detach().cpu().numpy() for centers_i in enrc_module.centers]
    V = enrc_module.V.detach().cpu().numpy()
    betas = enrc_module.subspace_betas().detach().cpu().numpy()
    P = enrc_module.P
    m = enrc_module.m
    return cluster_labels, cluster_centers, V, m, betas, P, n_clusters, autoencoder, cluster_labels_before_reclustering
