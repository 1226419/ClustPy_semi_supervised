import torch
from clustpy.deep.acedec_init import acedec_init


def acedec_recluster(y, embedded_data, n_clusters, init="auto", rounds=10, input_centers=None, P=None, V=None,
                random_state=None,
                max_iter=100, learning_rate=None, optimizer_class=None, batch_size=128, epochs=10,
                device=torch.device("cpu"),  debug=True, init_kwargs=None, reclustering="like_init",
                reclustering_kwargs=None):

    if reclustering == "like_init" or reclustering == "acedec":
        centers_reclustered, P, new_V, beta_weights = acedec_init(y=y, embedded_data=embedded_data,
                                                    n_clusters=n_clusters,
                                                    device=device, init=init,
                                                    rounds=rounds, epochs=epochs, debug=debug,
                                                    #input_centers=input_centers,
                                                    max_iter=max_iter, learning_rate=learning_rate,
                                                    )
    elif reclustering == "labeled_reclustering":
        raise NotImplementedError
    else:
        raise ValueError(f"reclustering={reclustering} is not implemented.")
    return centers_reclustered, P, new_V, beta_weights