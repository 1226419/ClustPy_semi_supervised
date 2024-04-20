"""
===================== Initialization Strategies =====================
"""
import torch
from sklearn.cluster import KMeans
import numpy as np
from clustpy.deep._data_utils import get_dataloader
from clustpy.alternative import NrKmeans
from sklearn.utils import check_random_state
from clustpy.alternative.nrkmeans import _get_total_cost_function
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_module import _ENRC_Module
from clustpy.deep.semisupervised_enrc.helper_functions import _IdentityAutoencoder
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_init_betas import calculate_beta_weight
from scipy.spatial.distance import cdist


def available_init_strategies() -> list:
    """
    Returns a list of strings of available initialization strategies for ENRC and ACeDeC.
    At the moment following strategies are supported: nrkmeans, random, sgd, auto
    """
    return ['nrkmeans', 'random', 'sgd', 'auto', 'subkmeans', 'acedec']


def apply_init_function(data: np.ndarray, n_clusters: list, init: str = "auto", rounds: int = 10, input_centers: list = None,
              P: list = None, V: np.ndarray = None, random_state: np.random.RandomState = None, max_iter: int = 100,
              optimizer_params: dict = None, optimizer_class: torch.optim.Optimizer = None, batch_size: int = 128,
              epochs: int = 10, device: torch.device = torch.device("cpu"), debug: bool = True,
              init_kwargs: dict = None, clustering_module: torch.nn.Module = None) -> (list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy for the ENRC algorithm.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    init : str
        {'nrkmeans', 'random', 'sgd', 'auto'} or callable. Initialization strategies for parameters cluster_centers, V and beta of ENRC. (default='auto')

        'nrkmeans' : Performs the NrKmeans algorithm to get initial parameters. This strategy is preferred for small data sets,
        but the orthogonality constraint on V and subsequently for the clustered subspaces can be sometimes to limiting in practice,
        e.g., if clusterings in the data are not perfectly non-redundant.

        'random' : Same as 'nrkmeans', but max_iter is set to 10, so the performance is faster, but also less optimized, thus more random.

        'sgd' : Initialization strategy based on optimizing ENRC's parameters V and beta in isolation from the autoencoder using a mini-batch gradient descent optimizer.
        This initialization strategy scales better to large data sets than the 'nrkmeans' option and only constraints V using the reconstruction error (torch.nn.MSELoss),
        which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the 'sgd' strategy is that it can be less stable for small data sets.

        'auto' : Selects 'sgd' init if data.shape[0] > 100,000 or data.shape[1] > 1,000. For smaller data sets 'nrkmeans' init is used.

        If a callable is passed, it should take arguments data and n_clusters (additional parameters can be provided via the dictionary init_kwargs) and return an initialization (centers, P, V and beta_weights).

    rounds : int
        number of repetitions of the initialization procedure (default: 10)
    input_centers : list
        list of np.ndarray, optional parameter if initial cluster centers want to be set (optional) (default: None)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    random_state : np.random.RandomState
        random state for reproducible results (default: None)
    max_iter : int
        maximum number of iterations of NrKmeans.  Only used for init='nrkmeans' (default: 100)
    optimizer_params : dict
        parameters of the optimizer used to optimize V and beta, includes the learning rate. Only used for init='sgd'
    optimizer_class : torch.optim.Optimizer
        optimizer for training. If None then torch.optim.Adam will be used. Only used for init='sgd' (default: None)
    batch_size : int
        size of the data batches. Only used for init='sgd' (default: 128)
    epochs : int
        number of epochs for the actual clustering procedure. Only used for init='sgd' (default: 10)
    device : torch.device
        device on which should be trained on. Only used for init='sgd' (default: torch.device('cpu'))
    debug : bool
        if True then the cost of each round will be printed (default: True)
    init_kwargs : dict
        additional parameters that are used if init is a callable (optional) (default: None)
    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace
        list containing projections for each subspace
        orthogonal rotation matrix
        weights for softmax function to get beta values.

    Raises
    ----------
    ValueError : if init variable is passed that is not implemented.
    """
    if clustering_module is None:
        clustering_module = _ENRC_Module
    if debug:
        print("init params")
        print(init)
        print(init_kwargs)
        print(optimizer_params)
        print(optimizer_class)
    if init == "nrkmeans" or init == "subkmeans":
        centers, P, V, beta_weights = nrkmeans_init(data=data, n_clusters=n_clusters, rounds=rounds,
                                                    input_centers=input_centers, P=P, V=V, random_state=random_state,
                                                    debug=debug)
    elif init == "random":
        centers, P, V, beta_weights = random_nrkmeans_init(data=data, n_clusters=n_clusters, rounds=rounds,
                                                           input_centers=input_centers, P=P, V=V,
                                                           random_state=random_state, debug=debug)
    elif init == "sgd":
        centers, P, V, beta_weights = sgd_init(data=data, n_clusters=n_clusters, optimizer_params=optimizer_params,
                                               rounds=rounds, epochs=epochs, input_centers=input_centers, P=P, V=V,
                                               optimizer_class=optimizer_class, batch_size=batch_size,
                                               random_state=random_state, device=device, debug=debug,
                                               clustering_module=clustering_module)
    elif init == "acedec":
        centers, P, V, beta_weights = acedec_init(data=data, n_clusters=n_clusters, optimizer_params=optimizer_params,
                                                  rounds=rounds, epochs=epochs, input_centers=input_centers, P=P, V=V,
                                                  optimizer_class=optimizer_class, batch_size=batch_size,
                                                  random_state=random_state, device=device, debug=debug,
                                                  clustering_module=clustering_module)
    elif init == "auto":
        if data.shape[0] > 100000 or data.shape[1] > 1000:
            init = "sgd"
        else:
            init = "nrkmeans"
        centers, P, V, beta_weights = apply_init_function(data=data, n_clusters=n_clusters, device=device, init=init,
                                                rounds=rounds, input_centers=input_centers,
                                                P=P, V=V, random_state=random_state, max_iter=max_iter,
                                                optimizer_params=optimizer_params, optimizer_class=optimizer_class,
                                                epochs=epochs, debug=debug, clustering_module=clustering_module)
    elif callable(init):
        if init_kwargs is not None:
            centers, P, V, beta_weights = init(data, n_clusters, **init_kwargs)
        else:
            centers, P, V, beta_weights = init(data, n_clusters)
    else:
        raise ValueError(f"init={init} is not implemented.")
    return centers, P, V, beta_weights


def nrkmeans_init(data: np.ndarray, n_clusters: list, rounds: int = 10, max_iter: int = 100, input_centers: list = None,
                  P: list = None, V: np.ndarray = None, random_state: np.random.RandomState = None, debug=True) -> (
        list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on the NrKmeans Algorithm. This strategy is preferred for small data sets, but the orthogonality
    constraint on V and subsequently for the clustered subspaces can be sometimes to limiting in practice, e.g., if clusterings are
    not perfectly non-redundant.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    rounds : int
        number of repetitions of the NrKmeans algorithm (default: 10)
    max_iter : int
        maximum number of iterations of NrKmeans (default: 100)
    input_centers : list
        list of np.ndarray, optional parameter if initial cluster centers want to be set (optional) (default: None)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace
        list containing projections for each subspace
        orthogonal rotation matrix
        weights for softmax function to get beta values.
    """
    best = None
    lowest = np.inf
    if max(n_clusters) >= data.shape[1]:
        mdl_for_noisespace = True
        if debug:
            print("mdl_for_noisespace=True, because number of clusters is larger than data dimensionality")
    else:
        mdl_for_noisespace = False
    for i in range(rounds):
        nrkmeans = NrKmeans(n_clusters=n_clusters, cluster_centers=input_centers, P=P, V=V, max_iter=max_iter,
                            random_state=random_state, mdl_for_noisespace=mdl_for_noisespace)
        nrkmeans.fit(X=data)
        centers_i, P_i, V_i, scatter_matrices_i = nrkmeans.cluster_centers, nrkmeans.P, nrkmeans.V, nrkmeans.scatter_matrices_
        if len(P_i) != len(n_clusters):
            if debug:
                print(
                    f"WARNING: Lost Subspace. Found only {len(P_i)} subspaces for {len(n_clusters)} clusterings. Try to increase the size of the embedded space or the number of iterations of nrkmeans to avoid this from happening.")
        else:
            cost = _get_total_cost_function(V=V_i, P=P_i, scatter_matrices=scatter_matrices_i)
            if lowest > cost:
                best = [centers_i, P_i, V_i, ]
                lowest = cost
            if debug:
                print(f"Round {i}: Found solution with: {cost} (current best: {lowest})")

    # Best parameters
    if best is None:
        centers, P, V = centers_i, P_i, V_i
        if debug:
            print(
                f"WARNING: No result with all subspaces was found. Will return last computed result with {len(P)} subspaces.")
    else:
        centers, P, V = best
    # centers are expected to be rotated for ENRC
    centers = [np.matmul(centers_sub, V) for centers_sub in centers]
    beta_weights = calculate_beta_weight(data=torch.from_numpy(data).float(),
                                         centers=[torch.from_numpy(centers_sub).float() for centers_sub in centers],
                                         V=torch.from_numpy(V).float(),
                                         P=P)
    beta_weights = beta_weights.detach().cpu().numpy()

    return centers, P, V, beta_weights


def random_nrkmeans_init(data: np.ndarray, n_clusters: list, rounds: int = 10, input_centers: list = None,
                         P: list = None, V: np.ndarray = None, random_state: np.random.RandomState = None,
                         debug: bool = True) -> (list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on the NrKmeans Algorithm. For documentation see nrkmeans_init function.
    Same as nrkmeans_init, but max_iter is set to 1, so the results will be faster and more random.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    rounds : int
        number of repetitions of the NrKmeans algorithm (default: 10)
    input_centers : list
        list of np.ndarray, optional parameter if initial cluster centers want to be set (optional) (default: None)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace
        list containing projections for each subspace
        orthogonal rotation matrix
        weights for softmax function to get beta values.
    """
    return nrkmeans_init(data=data, n_clusters=n_clusters, rounds=rounds, max_iter=1,
                         input_centers=input_centers, P=P, V=V, random_state=random_state, debug=debug)


def _determine_sgd_init_costs(enrc: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                              loss_fn: torch.nn.modules.loss._Loss, device: torch.device,
                              return_rot: bool = False) -> float:
    """
    Determine the initial sgd costs.

    Parameters
    ----------
    enrc : a torch.nn.Module based on _ENRC_Module
        The module
    dataloader : torch.utils.data.DataLoader
        dataloader to be used for the calculation of the costs
    loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction
    device : torch.device
        device to be trained on
    return_rot : bool
        if True rotated data from datalaoder will be returned (default: False)

    Returns
    -------
    cost : float
        the costs
    """
    cost = 0
    rotated_data = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[1].to(device)
            subspace_loss, z_rot, batch_rot_back, _ = enrc(batch)
            rotated_data.append(z_rot.detach().cpu())
            rec_loss = loss_fn(batch_rot_back, batch)
            cost += (subspace_loss + rec_loss)
        cost /= len(dataloader)
    if return_rot:
        rotated_data = torch.cat(rotated_data).numpy()
        return cost.item(), rotated_data
    else:
        return cost.item()


def sgd_init(data: np.ndarray, n_clusters: list, optimizer_params: dict, batch_size: int = 128,
             optimizer_class: torch.optim.Optimizer = None, rounds: int = 2, epochs: int = 10,
             random_state: np.random.RandomState = None, input_centers: list = None, P: list = None,
             V: np.ndarray = None, device: torch.device = torch.device("cpu"), debug: bool = True,
             clustering_module: torch.nn.Module = None) -> (list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on optimizing ENRC's parameters V and beta in isolation from the autoencoder using a mini-batch gradient descent optimizer.
    This initialization strategy scales better to large data sets than the nrkmeans_init and only constraints V using the reconstruction error (torch.nn.MSELoss),
    which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the sgd_init strategy is that it can be less stable for small data sets.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    optimizer_params : dict
        parameters of the optimizer used to optimize V and beta, includes the learning rate
    batch_size : int
        size of the data batches (default: 128)
    optimizer_class : torch.optim.Optimizer
        optimizer for training. If None then torch.optim.Adam will be used (default: None)
    rounds : int
        number of repetitions of the initialization procedure (default: 2)
    epochs : int
        number of epochs for the actual clustering procedure (default: 10)
    random_state : np.random.RandomState
        random state for reproducible results (default: None)
    input_centers : list
        list of np.ndarray, default=None, optional parameter if initial cluster centers want to be set (optional)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    device : torch.device
        device on which should be trained on (default: torch.device('cpu'))
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace,
        list containing projections for each subspace,
        orthogonal rotation matrix,
        weights for softmax function to get beta values.
    """
    best = None
    lowest = np.inf
    dataloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    for round_i in range(rounds):
        random_state = check_random_state(random_state)
        # start with random initialization
        init_centers, P_init, V_init, _ = random_nrkmeans_init(data=data, n_clusters=n_clusters, rounds=10,
                                                               input_centers=input_centers,
                                                               P=P, V=V, debug=debug)

        # Initialize betas with uniform distribution
        enrc_module = clustering_module(init_centers, P_init, V_init, beta_init_value=1.0 / len(P_init)).to_device(device)
        enrc_module.to_device(device)
        optimizer_beta_params = optimizer_params.copy()
        optimizer_beta_params["lr"] = optimizer_beta_params["lr"] * 10
        param_dict = [dict({'params': [enrc_module.V]}, **optimizer_params),
                      dict({'params': [enrc_module.beta_weights]}, **optimizer_beta_params)
                      ]
        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
        optimizer = optimizer_class(param_dict)
        # Training loop
        # For the initialization we increase the weight for the rec error to enforce close to orthogonal V by setting fix_rec_error=True
        enrc_module.fit(data=data,
                        trainloader=None,
                        evalloader=None,
                        optimizer=optimizer,
                        max_epochs=epochs,
                        model=_IdentityAutoencoder(),
                        loss_fn=torch.nn.MSELoss(),
                        batch_size=batch_size,
                        device=device,
                        debug=False,
                        fix_rec_error=True)

        cost = _determine_sgd_init_costs(enrc=enrc_module, dataloader=dataloader, loss_fn=torch.nn.MSELoss(),
                                         device=device)
        if lowest > cost:
            best = [enrc_module.centers, enrc_module.P, enrc_module.V, enrc_module.beta_weights]
            lowest = cost
        if debug:
            print(f"Round {round_i}: Found solution with: {cost} (current best: {lowest})")

    centers, P, V, beta_weights = best
    beta_weights = calculate_beta_weight(data=torch.from_numpy(data).float(), centers=centers, V=V, P=P)
    centers = [centers_i.detach().cpu().numpy() for centers_i in centers]
    beta_weights = beta_weights.detach().cpu().numpy()
    V = V.detach().cpu().numpy()
    return centers, P, V, beta_weights


def acedec_init(data: np.ndarray, n_clusters: list, optimizer_params: dict, batch_size: int = 128,
                optimizer_class: torch.optim.Optimizer = None, rounds: int = None, epochs: int = 10,
                random_state: np.random.RandomState = None, input_centers: list = None, P: list = None,
                V: np.ndarray = None, device: torch.device = torch.device("cpu"), debug: bool = True,
                clustering_module: torch.nn.Module = None) -> (
        list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on optimizing ACeDeC's parameters V and beta in isolation from the autoencoder using a mini-batch gradient descent optimizer.
    This initialization strategy scales better to large data sets than the nrkmeans_init and only constraints V using the reconstruction error (torch.nn.MSELoss),
    which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the sgd_init strategy is that it can be less stable for small data sets.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    optimizer_params : dict
        parameters of the optimizer used to optimize V and beta, includes the learning rate
    batch_size : int
        size of the data batches (default: 128)
    optimizer_params: dict
            parameters of the optimizer for the actual clustering procedure, includes the learning rate
    optimizer_class : torch.optim.Optimizer
        optimizer for training. If None then torch.optim.Adam will be used (default: None)
    rounds : int
        not used here (default: None)
    epochs : int
        epochs is automatically set to be close to 20.000 minibatch iterations as in the ACeDeC paper. If this determined value is smaller than the passed epochs, then epochs is used (default: 10)
    random_state : np.random.RandomState
        random state for reproducible results (default: None)
    input_centers : list
        list of np.ndarray, default=None, optional parameter if initial cluster centers want to be set (optional)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    device : torch.device
        device on which should be trained on (default: torch.device('cpu'))
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace,
        list containing projections for each subspace,
        orthogonal rotation matrix,
        weights for softmax function to get beta values.
    """
    best = None
    lowest = np.inf
    dataloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    # only use one repeat as in ACeDeC paper
    acedec_rounds = 1
    # acedec used 20.000 minibatch iterations for initialization. Thus we use a number of epochs corresponding to that

    epochs_estimate = int(20000 / (data.shape[0] / batch_size))
    max_epochs = np.max([epochs_estimate, epochs])

    if debug: print("Start ACeDeC init")
    for round_i in range(acedec_rounds):
        random_state = check_random_state(random_state)

        # start with random initialization
        if debug: print("Start with random init")
        init_centers, P_init, V_init, _ = random_nrkmeans_init(data=data, n_clusters=n_clusters, rounds=10,
                                                               input_centers=input_centers,
                                                               P=P, V=V, debug=debug)
        # Recluster with KMeans to get better centroid estimate
        data_rot = np.matmul(data, V_init)
        kmeans = KMeans(n_clusters[0], n_init=10)
        kmeans.fit(data_rot)
        # cluster and shared space centers
        init_centers = [kmeans.cluster_centers_, data_rot.mean(0).reshape(1, -1)]

        # Initialize betas with uniform distribution
        enrc_module = clustering_module(init_centers, P_init, V_init, beta_init_value=1.0 / len(P_init)).to_device(device)
        enrc_module.to_device(device)

        optimizer_beta_params = optimizer_params.copy()
        optimizer_beta_params["lr"] = optimizer_beta_params["lr"] * 10
        param_dict = [dict({'params': [enrc_module.V]}, **optimizer_params),
                      dict({'params': [enrc_module.beta_weights]}, **optimizer_beta_params)
                      ]
        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
        optimizer = optimizer_class(param_dict)
        # Training loop
        # For the initialization we increase the weight for the rec error to enforce close to orthogonal V by setting fix_rec_error=True
        if debug:
            print("Start pretraining parameters with SGD")
            print("data shape", data.shape)
            print("max_epochs", max_epochs)
            print("batch_size", batch_size)
        enrc_module.fit(data=data,
                        trainloader=None,
                        evalloader=None,
                        optimizer=optimizer,
                        max_epochs=max_epochs,
                        model=_IdentityAutoencoder(),
                        loss_fn=torch.nn.MSELoss(),
                        batch_size=batch_size,
                        device=device,
                        debug=debug,
                        fix_rec_error=True)

        cost, z_rot = _determine_sgd_init_costs(enrc=enrc_module, dataloader=dataloader, loss_fn=torch.nn.MSELoss(),
                                                device=device, return_rot=True)

        # Recluster with KMeans to get better centroid estimate
        kmeans = KMeans(n_clusters[0], n_init=10)
        kmeans.fit(z_rot)
        # cluster and shared space centers
        enrc_rotated_centers = [kmeans.cluster_centers_, z_rot.mean(0).reshape(1, -1)]
        enrc_module.centers = [torch.tensor(centers_sub, dtype=torch.float32) for centers_sub in enrc_rotated_centers]

        if lowest > cost:
            best = [enrc_module.centers, enrc_module.P, enrc_module.V, enrc_module.beta_weights]
            lowest = cost
        if debug:
            print(f"Round {round_i}: Found solution with: {cost} (current best: {lowest})")

    centers, P, V, beta_weights = best

    beta_weights = calculate_beta_weight(data=torch.from_numpy(data).float(), centers=centers, V=V, P=P)
    centers = [centers_i.detach().cpu().numpy() for centers_i in centers]
    beta_weights = beta_weights.detach().cpu().numpy()
    V = V.detach().cpu().numpy()
    return centers, P, V, beta_weights


def semi_supervised_acedec_init(data: np.ndarray, n_clusters: list, optimizer_params: dict = None, batch_size: int = 128,
                optimizer_class: torch.optim.Optimizer = None, rounds: int = None, epochs: int = 10,
                random_state: np.random.RandomState = None, input_centers: list = None, P: list = None,
                V: np.ndarray = None, device: torch.device = torch.device("cpu"), debug: bool = True,
                clustering_module: torch.nn.Module = _ENRC_Module, y: np.ndarray = None) -> (
        list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on optimizing ACeDeC's parameters V and beta in isolation from the autoencoder using a mini-batch gradient descent optimizer.
    This initialization strategy scales better to large data sets than the nrkmeans_init and only constraints V using the reconstruction error (torch.nn.MSELoss),
    which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the sgd_init strategy is that it can be less stable for small data sets.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    optimizer_params : dict
        parameters of the optimizer used to optimize V and beta, includes the learning rate
    batch_size : int
        size of the data batches (default: 128)
    optimizer_params: dict
            parameters of the optimizer for the actual clustering procedure, includes the learning rate
    optimizer_class : torch.optim.Optimizer
        optimizer for training. If None then torch.optim.Adam will be used (default: None)
    rounds : int
        not used here (default: None)
    epochs : int
        epochs is automatically set to be close to 20.000 minibatch iterations as in the ACeDeC paper. If this determined value is smaller than the passed epochs, then epochs is used (default: 10)
    random_state : np.random.RandomState
        random state for reproducible results (default: None)
    input_centers : list
        list of np.ndarray, default=None, optional parameter if initial cluster centers want to be set (optional)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    device : torch.device
        device on which should be trained on (default: torch.device('cpu'))
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace,
        list containing projections for each subspace,
        orthogonal rotation matrix,
        weights for softmax function to get beta values.
    """
    print("semisupervised_init")
    if input_centers is None:
        if int(sum(y)) == len(y) * -1:
            print("No labels available - falling back to unsupervised initialization")
            centers, P, V, beta_weights = acedec_init(data=data, n_clusters=n_clusters, optimizer_params=optimizer_params,
                                                  rounds=rounds, epochs=epochs, input_centers=input_centers, P=P, V=V,
                                                  optimizer_class=optimizer_class, batch_size=batch_size,
                                                  random_state=random_state, device=device, debug=debug,
                                                  clustering_module=clustering_module)
        else:
            print("Labels available - performing supervised initialization")
            input_centers = []
            # calculate centers for each subspace using the labels
            input_centers = calculate_centers_from_labels(y, data, n_clusters, input_centers)

            # calculate centers for the noise subspace
            noice_center = np.mean(data, axis=0)
            input_centers.append(torch.from_numpy(np.array([noice_center])))

            best = None
            lowest = np.inf
            dataloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)
            # only use one repeat as in ACeDeC paper
            # acedec used 20.000 minibatch iterations for initialization. Thus we use a number of epochs corresponding to that

            epochs_estimate = int(20000 / (data.shape[0] / batch_size))
            max_epochs = np.max([epochs_estimate, epochs])


            # start with labeled initialization
            if debug: print("Start with random init using the centers from the labels")

            init_centers, P_init, V_init, _ = random_nrkmeans_init(data=data, n_clusters=n_clusters, rounds=10,
                                                                   input_centers=input_centers,
                                                                   P=P, V=V, debug=debug)
            print("huh")
            # Recluster with KMeans to get better centroid estimate
            data_rot = np.matmul(data, V_init)
            kmeans = KMeans(n_clusters[0], n_init=10)
            kmeans.fit(data_rot)
            # cluster and shared space centers
            init_centers = [kmeans.cluster_centers_, data_rot.mean(0).reshape(1, -1)]
            print(init_centers)
            print(P_init)
            print(V_init)
            print(clustering_module)
            # Initialize betas with uniform distribution
            enrc_module = clustering_module(init_centers, P_init, V_init,
                                            beta_init_value=1.0 / len(P_init)).to_device(device)
            enrc_module.to_device(device)

            optimizer_beta_params = optimizer_params.copy()
            optimizer_beta_params["lr"] = optimizer_beta_params["lr"] * 10
            param_dict = [dict({'params': [enrc_module.V]}, **optimizer_params),
                          dict({'params': [enrc_module.beta_weights]}, **optimizer_beta_params)
                          ]
            if optimizer_class is None:
                optimizer_class = torch.optim.Adam
            optimizer = optimizer_class(param_dict)
            # Training loop
            # For the initialization we increase the weight for the rec error to enforce close to orthogonal V by setting fix_rec_error=True
            if debug:
                print("Start pretraining parameters with SGD")
                print("data shape", data.shape)
                print("max_epochs", max_epochs)
                print("batch_size", batch_size)
            enrc_module.fit(data=data,
                            trainloader=None,
                            evalloader=None,
                            optimizer=optimizer,
                            max_epochs=max_epochs,
                            model=_IdentityAutoencoder(),
                            loss_fn=torch.nn.MSELoss(),
                            batch_size=batch_size,
                            device=device,
                            debug=debug,
                            fix_rec_error=True)

            cost, z_rot = _determine_sgd_init_costs(enrc=enrc_module, dataloader=dataloader,
                                                    loss_fn=torch.nn.MSELoss(),
                                                    device=device, return_rot=True)

            # Recluster with KMeans to get better centroid estimate
            kmeans = KMeans(n_clusters[0], n_init=10)
            kmeans.fit(z_rot)
            # cluster and shared space centers
            enrc_rotated_centers = [kmeans.cluster_centers_, z_rot.mean(0).reshape(1, -1)]
            enrc_module.centers = [torch.tensor(centers_sub, dtype=torch.float32) for centers_sub in
                                   enrc_rotated_centers]

            if lowest > cost:
                best = [enrc_module.centers, enrc_module.P, enrc_module.V, enrc_module.beta_weights]
                lowest = cost
            if debug:
                print(f"Found solution with: {cost} (current best: {lowest})")
            centers, P, V, beta_weights = best
            beta_weights = calculate_beta_weight(data=torch.from_numpy(data).float(), centers=centers, V=V, P=P)
            centers = [centers_i.detach().cpu().numpy() for centers_i in centers]
            beta_weights = beta_weights.detach().cpu().numpy()
            V = V.detach().cpu().numpy()

    return centers, P, V, beta_weights


def semi_supervised_acedec_init_simple(data: np.ndarray, n_clusters: list, optimizer_params: dict = None, batch_size: int = 128,
                optimizer_class: torch.optim.Optimizer = None, rounds: int = None, epochs: int = 10,
                random_state: np.random.RandomState = None, input_centers: list = None, P: list = None,
                V: np.ndarray = None, device: torch.device = torch.device("cpu"), debug: bool = True,
                clustering_module: torch.nn.Module = _ENRC_Module, y: np.ndarray = None) -> (
        list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on optimizing ACeDeC's parameters V and beta in isolation from the autoencoder using a mini-batch gradient descent optimizer.
    This initialization strategy scales better to large data sets than the nrkmeans_init and only constraints V using the reconstruction error (torch.nn.MSELoss),
    which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the sgd_init strategy is that it can be less stable for small data sets.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    optimizer_params : dict
        parameters of the optimizer used to optimize V and beta, includes the learning rate
    batch_size : int
        size of the data batches (default: 128)
    optimizer_params: dict
            parameters of the optimizer for the actual clustering procedure, includes the learning rate
    optimizer_class : torch.optim.Optimizer
        optimizer for training. If None then torch.optim.Adam will be used (default: None)
    rounds : int
        not used here (default: None)
    epochs : int
        epochs is automatically set to be close to 20.000 minibatch iterations as in the ACeDeC paper. If this determined value is smaller than the passed epochs, then epochs is used (default: 10)
    random_state : np.random.RandomState
        random state for reproducible results (default: None)
    input_centers : list
        list of np.ndarray, default=None, optional parameter if initial cluster centers want to be set (optional)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    device : torch.device
        device on which should be trained on (default: torch.device('cpu'))
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace,
        list containing projections for each subspace,
        orthogonal rotation matrix,
        weights for softmax function to get beta values.
    """
    print("semisupervised_init")
    if input_centers is None:
        if int(sum(y)) == len(y) * -1:
            print("No labels available - falling back to unsupervised initialization")
            centers, P, V, beta_weights = acedec_init(data=data, n_clusters=n_clusters, optimizer_params=optimizer_params,
                                                  rounds=rounds, epochs=epochs, input_centers=input_centers, P=P, V=V,
                                                  optimizer_class=optimizer_class, batch_size=batch_size,
                                                  random_state=random_state, device=device, debug=debug,
                                                  clustering_module=clustering_module)
        else:
            print("Labels available - performing supervised initialization simple version")
            input_centers = []
            # calculate centers for each subspace using the labels
            input_centers = calculate_centers_from_labels(y, data, n_clusters, input_centers)
            # calculate centers for the noise subspace
            noice_center = np.mean(data, axis=0)
            input_centers.append(torch.from_numpy(np.array([noice_center])))

            best = None
            lowest = np.inf
            dataloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)
            # only use one repeat as in ACeDeC paper
            # acedec used 20.000 minibatch iterations for initialization. Thus we use a number of epochs corresponding to that

            epochs_estimate = int(20000 / (data.shape[0] / batch_size))
            max_epochs = np.max([epochs_estimate, epochs])


            # start with labeled initialization
            if debug: print("Start with random init using the centers from the labels")

            init_centers = input_centers
            P_init = None # TODO
            m = None
            # Get number of subspaces
            subspaces = len(n_clusters)
            data_dimensionality = data.shape[1]
            # Calculate dimensionalities m
            if m is None and P_init is None:
                m = [int(data_dimensionality / subspaces)] * subspaces
                if data_dimensionality % subspaces != 0:
                    choices = random_state.choice(subspaces, data_dimensionality - np.sum(m), replace=False)
                    for choice in choices:
                        m[choice] += 1
            if P_init is None:
                possible_projections = list(range(data_dimensionality))
                P_init = []
                if random_state is None:
                    random_state = np.random.RandomState()
                for dimensionality in m:
                    choices = random_state.choice(possible_projections, dimensionality, replace=False)
                    P_init.append(choices)
                    possible_projections = list(set(possible_projections) - set(choices))
            V_init = np.eye(data.shape[1])

            # Initialize betas with uniform distribution
            enrc_module = clustering_module(init_centers, P_init, V_init,
                                            beta_init_value=1.0 / len(P_init)).to_device(device)
            enrc_module.to_device(device)

            optimizer_beta_params = optimizer_params.copy()
            optimizer_beta_params["lr"] = optimizer_beta_params["lr"] * 10
            param_dict = [dict({'params': [enrc_module.V]}, **optimizer_params),
                          dict({'params': [enrc_module.beta_weights]}, **optimizer_beta_params)
                          ]
            if optimizer_class is None:
                optimizer_class = torch.optim.Adam
            optimizer = optimizer_class(param_dict)
            # Training loop
            # For the initialization we increase the weight for the rec error to enforce close to orthogonal V by setting fix_rec_error=True
            if debug:
                print("Start pretraining parameters with SGD")
                print("data shape", data.shape)
                print("max_epochs", max_epochs)
                print("batch_size", batch_size)
            enrc_module.fit(data=data,
                            trainloader=None,
                            evalloader=None,
                            optimizer=optimizer,
                            max_epochs=max_epochs,
                            model=_IdentityAutoencoder(),
                            loss_fn=torch.nn.MSELoss(),
                            batch_size=batch_size,
                            device=device,
                            debug=debug,
                            fix_rec_error=False)

            cost, z_rot = _determine_sgd_init_costs(enrc=enrc_module, dataloader=dataloader,
                                                    loss_fn=torch.nn.MSELoss(),
                                                    device=device, return_rot=True)

            # Recluster with KMeans to get better centroid estimate
            kmeans = KMeans(n_clusters[0], n_init=10)
            kmeans.fit(z_rot)
            # cluster and shared space centers
            enrc_rotated_centers = [kmeans.cluster_centers_, z_rot.mean(0).reshape(1, -1)]
            enrc_module.centers = [torch.tensor(centers_sub, dtype=torch.float32) for centers_sub in
                                   enrc_rotated_centers]

            if lowest > cost:
                best = [enrc_module.centers, enrc_module.P, enrc_module.V, enrc_module.beta_weights]
                lowest = cost
            if debug:
                print(f"Found solution with: {cost} (current best: {lowest})")
            centers, P, V, beta_weights = best
            beta_weights = calculate_beta_weight(data=torch.from_numpy(data).float(), centers=centers, V=V, P=P)
            centers = [centers_i.detach().cpu().numpy() for centers_i in centers]
            beta_weights = beta_weights.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
            print("simple init done")
            print(centers)
            print(P)
            print(V)
            print(beta_weights)
    return centers, P, V, beta_weights


def semi_supervised_acedec_init_simplest(data: np.ndarray, n_clusters: list, optimizer_params: dict = None, batch_size: int = 128,
                optimizer_class: torch.optim.Optimizer = None, rounds: int = None, epochs: int = 10,
                random_state: np.random.RandomState = None, input_centers: list = None, P: list = None,
                V: np.ndarray = None, device: torch.device = torch.device("cpu"), debug: bool = True,
                clustering_module: torch.nn.Module = _ENRC_Module, y: np.ndarray = None) -> (
        list, list, np.ndarray, np.ndarray):
    """
    Initialization strategy based on optimizing ACeDeC's parameters V and beta in isolation from the autoencoder using a mini-batch gradient descent optimizer.
    This initialization strategy scales better to large data sets than the nrkmeans_init and only constraints V using the reconstruction error (torch.nn.MSELoss),
    which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the sgd_init strategy is that it can be less stable for small data sets.

    Parameters
    ----------
    data : np.ndarray
        input data
    n_clusters : list
        list of ints, number of clusters for each clustering
    optimizer_params : dict
        parameters of the optimizer used to optimize V and beta, includes the learning rate
    batch_size : int
        size of the data batches (default: 128)
    optimizer_params: dict
            parameters of the optimizer for the actual clustering procedure, includes the learning rate
    optimizer_class : torch.optim.Optimizer
        optimizer for training. If None then torch.optim.Adam will be used (default: None)
    rounds : int
        not used here (default: None)
    epochs : int
        epochs is automatically set to be close to 20.000 minibatch iterations as in the ACeDeC paper. If this determined value is smaller than the passed epochs, then epochs is used (default: 10)
    random_state : np.random.RandomState
        random state for reproducible results (default: None)
    input_centers : list
        list of np.ndarray, default=None, optional parameter if initial cluster centers want to be set (optional)
    P : list
        list containing projections for each subspace (optional) (default: None)
    V : np.ndarray
        orthogonal rotation matrix (optional) (default: None)
    device : torch.device
        device on which should be trained on (default: torch.device('cpu'))
    debug : bool
        if True then the cost of each round will be printed (default: True)

    Returns
    -------
    tuple : (list, list, np.ndarray, np.ndarray)
        list of cluster centers for each subspace,
        list containing projections for each subspace,
        orthogonal rotation matrix,
        weights for softmax function to get beta values.
    """
    print("semisupervised_init simplest")
    if random_state is None:
        random_state = np.random.RandomState()
    if input_centers is None:
        if int(sum(y)) == len(y) * -1:
            print("No labels available - falling back to unsupervised initialization")
            centers, P_init, V_init, beta_weights = acedec_init(data=data, n_clusters=n_clusters, optimizer_params=optimizer_params,
                                                  rounds=rounds, epochs=epochs, input_centers=input_centers, P=P, V=V,
                                                  optimizer_class=optimizer_class, batch_size=batch_size,
                                                  random_state=random_state, device=device, debug=debug,
                                                  clustering_module=clustering_module)
        else:
            print("Labels available - performing supervised initialization simple version")
            input_centers = []

            # calculate centers for each subspace using the labels
            input_centers = calculate_centers_from_labels(y, data, n_clusters, input_centers)

            # calculate centers for the noise subspace
            noice_center = np.mean(data, axis=0)
            input_centers.append(torch.from_numpy(np.array([noice_center])))

            # start with labeled initialization
            if debug: print("Start with random init using the centers from the labels")

            P_init = None
            m = None
            # Get number of subspaces
            subspaces = len(n_clusters)
            data_dimensionality = data.shape[1]
            # Calculate dimensionalities m
            if m is None and P_init is None:
                m = [int(data_dimensionality / subspaces)] * subspaces
                if data_dimensionality % subspaces != 0:
                    choices = random_state.choice(subspaces, data_dimensionality - np.sum(m), replace=False)
                    for choice in choices:
                        m[choice] += 1
            if P_init is None:
                possible_projections = list(range(data_dimensionality))
                P_init = []
                for dimensionality in m:
                    choices = random_state.choice(possible_projections, dimensionality, replace=False)
                    P_init.append(choices)
                    possible_projections = list(set(possible_projections) - set(choices))
            V_init = torch.eye(data.shape[1]).to(device)
            centers = input_centers
            beta_weights = calculate_beta_weight(data=torch.from_numpy(data).float(), centers=centers,V=V_init, P=P)

            centers = [centers_i.detach().cpu().numpy() for centers_i in centers]
            beta_weights = beta_weights.detach().cpu().numpy()
            V_init = V_init.detach().cpu().numpy()

    return centers, P_init, V_init, beta_weights


def calculate_centers_from_labels(y, data, n_clusters: list, input_centers: list):
    for subspace_size in n_clusters[:-1]:
        subspace_centers = []
        labels = np.unique(y)
        if -1 in labels:
            labels = labels[1:]
        for label_index in labels:
            current_label_mask = np.where(y == label_index)[0]
            tmp = data[current_label_mask]
            tmp_mean = np.mean(tmp, axis=0)
            subspace_centers.append(tmp_mean)
        subspace_centers = np.array(subspace_centers)
        print(subspace_centers.shape)
        nr_cluster_without_label = subspace_size - len(labels)
        for i in range(nr_cluster_without_label):
            print("not all labels occur in label data "
                  "-> find unlabeled point in data that has the highest minimum distance "
                  "to all cluster_centers and set as new initial cluster center")
            current_label_mask = np.where(y == -1)[0]
            tmp = data[current_label_mask]
            distance_matrix = cdist(tmp, subspace_centers)
            min_distances = np.min(distance_matrix, axis=1)
            row_index = np.argmax(min_distances)
            row_with_highest_min_distance = tmp[row_index]
            subspace_centers = np.vstack((subspace_centers, row_with_highest_min_distance))
        input_centers.append(torch.from_numpy(subspace_centers))
    return input_centers
