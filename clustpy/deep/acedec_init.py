import torch
import numpy as np
from clustpy.deep.enrc import enrc_init, available_init_strategies
from clustpy.utils.plots import plot_scatter_matrix, plot_2d_data

def acedec_init(y, embedded_data, n_clusters, init="auto", rounds=10, input_centers=None, P=None, V=None,
                random_state=None,
                max_iter=100, learning_rate=None, optimizer_class=None, batch_size=128, epochs=10,
                device=torch.device("cpu"),  debug=True, init_kwargs=None):
    """Initialization strategy for the ENRC algorithm.

        Parameters
        ----------
        data : np.ndarray, input data
        n_clusters : list of ints, number of clusters for each clustering
        init : {'nrkmeans', 'random', 'sgd', 'auto'} or callable, default='auto'. Initialization strategies for parameters cluster_centers, V and beta of ENRC.

            'nrkmeans' : Performs the NrKmeans algorithm to get initial parameters. This strategy is preferred for small data sets,
            but the orthogonality constraint on V and subsequently for the clustered subspaces can be sometimes to limiting in practice,
            e.g., if clusterings in the data are not perfectly non-redundant.

            'random' : Same as 'nrkmeans', but max_iter is set to 10, so the performance is faster, but also less optimized, thus more random.

            'sgd' : Initialization strategy based on optimizing ENRC's parameters V and beta in isolation from the autoencoder using a mini-batch gradient descent optimizer.
            This initialization strategy scales better to large data sets than the 'nrkmeans' option and only constraints V using the reconstruction error (torch.nn.MSELoss),
            which can be more flexible than the orthogonality constraint of NrKmeans. A problem of the 'sgd' strategy is that it can be less stable for small data sets.

            'auto' : Selects 'sgd' init if data.shape[0] > 100,000 or data.shape[1] > 1,000. For smaller data sets 'nrkmeans' init is used.

            If a callable is passed, it should take arguments data and n_clusters (additional parameters can be provided via the dictionary init_kwargs) and return an initialization (centers, P, V and beta_weights).
        X : np.ndarray, input data
        y : np.ndarray, input labels
        rounds : int, default=10, number of repetitions of the initialization procedure
        input_centers : list of np.ndarray, default=None, optional parameter if initial cluster centers want to be set (optional)
        V : np.ndarray, default=None, orthogonal rotation matrix (optional)
        P : list, default=None, list containing projections for each subspace (optional)
        random_state : int, default=None, random state for reproducible results
        max_iter : int, default=100, maximum number of iterations of NrKmeans.  Only used for init='nrkmeans'.
        learning_rate : float, learning rate for optimizer_class that is used to optimize V and beta. Only used for init='sgd'.
        optimizer_class : torch.optim, default=None, optimizer for training. If None then torch.optim.Adam will be used. Only used for init='sgd'.
        batch_size : int, default=128, size of the data batches. Only used for init='sgd'.
        epochs : int, default=10, number of epochs for the actual clustering procedure. Only used for init='sgd'.
        device : torch.device, default=torch.device('cpu'), device on which should be trained on. Only used for init='sgd'.
        debug : bool, default=True, if True then the cost of each round will be printed.
        init_kwargs : dict, default=None, additional parameters that are used if init is a callable. (optional)

        Returns
        -------
        centers : list of cluster centers for each subspace
        P : list containing projections for each subspace
        V : np.ndarray, orthogonal rotation matrix
        beta_weights: np.ndarray, weights for softmax function to get beta values.

        Raises
        ----------
        ValueError : if init variable is passed that is not implemented.
        """
    if init in available_init_strategies() or callable(init):
        input_centers, P, V, beta_weights = enrc_init(data=embedded_data, n_clusters=n_clusters,
                                                      device=device, init=init,
                                                      rounds=rounds, epochs=epochs, batch_size=batch_size, debug=debug,
                                                      input_centers=input_centers, P=P, V=V, random_state=random_state,
                                                      max_iter=max_iter, optimizer_params={"lr": learning_rate},
                                                      optimizer_class=optimizer_class, init_kwargs=init_kwargs)
    elif init == "acedec_labeled":
        input_centers, P, V, beta_weights = acedec_labeled_init(data=embedded_data, n_clusters=n_clusters,
                                                      device=device, init=init,
                                                      rounds=rounds, epochs=epochs, batch_size=batch_size, debug=debug,
                                                      input_centers=input_centers, P=P, V=V, random_state=random_state,
                                                      max_iter=max_iter, optimizer_params={"lr": learning_rate},
                                                      optimizer_class=optimizer_class, init_kwargs=init_kwargs)
    else:
        raise ValueError(f"init={init} is not implemented.")

    if input_centers is not None and debug:
        plot_2d_data(embedded_data, y, input_centers[0], title="before enrc_init") # only plot first label dim (There
        # is only one for now.

    if debug:
        plot_2d_data(embedded_data@V, y, input_centers[0], title="after enrc_init")
    return input_centers, P, V, beta_weights


def acedec_labeled_init(y, embedded_data, n_clusters, init="auto", rounds=10, input_centers=None, P=None, V=None,
                random_state=None,
                max_iter=100, learning_rate=None, optimizer_class=None, batch_size=128, epochs=10,
                device=torch.device("cpu"),  debug=True, init_kwargs=None):
    # TODO check if the label_calculated_centers are changed too much in the enrc_init
    # TODO if multilabel betas can't be calculated and there is a slight difference in beta init from enrc and acedec
    if input_centers is None:
        if int(sum(y)) == len(y) * -1:
            print("No labels available - falling back to unsupervised initialization")
            input_centers = None
        else:
            print("Labels available - performing supervised initialization")
            input_centers = []
            # for subspace_size in n_clusters:
            #    input_centers.append(np.zeros((subspace_size, embedding_size)))
            # calculate centers for each subspace using the labels
            for subspace_size in n_clusters[:-1]:
                subspace_centers = []
                labels = np.unique(y)
                if -1 in labels:
                    labels = labels[1:]
                assert len(labels) == subspace_size
                for label_index in labels:
                    current_label_mask = np.where(y == label_index)[0]
                    if current_label_mask.shape[0] == 0:
                        # TODO: handle this case - initialise randomly
                        raise ValueError(f"Label {label_index} is not present in the data.")
                    tmp = embedded_data[current_label_mask]
                    tmp_mean = np.mean(tmp, axis=0)
                    subspace_centers.append(tmp_mean)
                subspace_centers = np.array(subspace_centers)
                input_centers.append(subspace_centers)

            # calculate centers for the noise subspace
            noice_center = np.mean(embedded_data, axis=0)
            input_centers.append([noice_center])
    input_centers, P, V, beta_weights = enrc_init(data=embedded_data, n_clusters=n_clusters,
                                                  device=device, init=init,
                                                  rounds=rounds, epochs=epochs, batch_size=batch_size, debug=debug,
                                                  input_centers=input_centers, P=P, V=V, random_state=random_state,
                                                  max_iter=max_iter, optimizer_params={"lr": learning_rate},
                                                  optimizer_class=optimizer_class, init_kwargs=init_kwargs)
    return input_centers, P, V, beta_weights