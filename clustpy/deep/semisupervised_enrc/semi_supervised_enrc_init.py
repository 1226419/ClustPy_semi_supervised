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


def available_init_strategies() -> list:
    """
    Returns a list of strings of available initialization strategies for ENRC and ACeDeC.
    At the moment following strategies are supported: nrkmeans, random, sgd, auto
    """
    return ['nrkmeans', 'random', 'sgd', 'auto', 'subkmeans', 'acedec']




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


def _determine_sgd_init_costs(enrc: _ENRC_Module, dataloader: torch.utils.data.DataLoader,
                              loss_fn: torch.nn.modules.loss._Loss, device: torch.device,
                              return_rot: bool = False) -> float:
    """
    Determine the initial sgd costs.

    Parameters
    ----------
    enrc : _ENRC_Module
        The ENRC module
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
             V: np.ndarray = None, device: torch.device = torch.device("cpu"), debug: bool = True) -> (
        list, list, np.ndarray, np.ndarray):
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
        enrc_module = _ENRC_Module(init_centers, P_init, V_init, beta_init_value=1.0 / len(P_init)).to_device(device)
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
                V: np.ndarray = None, device: torch.device = torch.device("cpu"), debug: bool = True) -> (
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
        enrc_module = _ENRC_Module(init_centers, P_init, V_init, beta_init_value=1.0 / len(P_init)).to_device(device)
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


def enrc_init(data: np.ndarray, n_clusters: list, init: str = "auto", rounds: int = 10, input_centers: list = None,
              P: list = None, V: np.ndarray = None, random_state: np.random.RandomState = None, max_iter: int = 100,
              optimizer_params: dict = None, optimizer_class: torch.optim.Optimizer = None, batch_size: int = 128,
              epochs: int = 10, device: torch.device = torch.device("cpu"), debug: bool = True,
              init_kwargs: dict = None) -> (list, list, np.ndarray, np.ndarray):
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
                                               random_state=random_state, device=device, debug=debug)
    elif init == "acedec":
        centers, P, V, beta_weights = acedec_init(data=data, n_clusters=n_clusters, optimizer_params=optimizer_params,
                                                  rounds=rounds, epochs=epochs, input_centers=input_centers, P=P, V=V,
                                                  optimizer_class=optimizer_class, batch_size=batch_size,
                                                  random_state=random_state, device=device, debug=debug)
    elif init == "auto":
        if data.shape[0] > 100000 or data.shape[1] > 1000:
            init = "sgd"
        else:
            init = "nrkmeans"
        centers, P, V, beta_weights = enrc_init(data=data, n_clusters=n_clusters, device=device, init=init,
                                                rounds=rounds, input_centers=input_centers,
                                                P=P, V=V, random_state=random_state, max_iter=max_iter,
                                                optimizer_params=optimizer_params, optimizer_class=optimizer_class,
                                                epochs=epochs, debug=debug)
    elif callable(init):
        if init_kwargs is not None:
            centers, P, V, beta_weights = init(data, n_clusters, **init_kwargs)
        else:
            centers, P, V, beta_weights = init(data, n_clusters)
    else:
        raise ValueError(f"init={init} is not implemented.")
    return centers, P, V, beta_weights
