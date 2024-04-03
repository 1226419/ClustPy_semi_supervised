import torch
import numpy as np
from clustpy.deep._utils import int_to_one_hot, squared_euclidean_distance
from clustpy.deep._data_utils import get_dataloader


def optimal_beta(kmeans_loss: torch.Tensor, other_losses_mean_sum: torch.Tensor) -> torch.Tensor:
    """
    Calculate optimal values for the beta weight for each dimension.

    Parameters
    ----------
    kmeans_loss: torch.Tensor
        a 1 x d vector of the kmeans losses per dimension.
    other_losses_mean_sum: torch.Tensor
        a 1 x d vector of the kmeans losses of all other clusterings except the one in 'kmeans_loss'.

    Returns
    -------
    optimal_beta_weights: torch.Tensor
        a 1 x d vector containing the optimal weights for the softmax to indicate which dimensions are important for each clustering.
        Calculated via -torch.log(kmeans_loss/other_losses_mean_sum)
    """
    return -torch.log(kmeans_loss / other_losses_mean_sum)


def calculate_optimal_beta_weights_special_case(data: torch.Tensor, centers: list, V: torch.Tensor,
                                                batch_size: int = 32) -> torch.Tensor:
    """
    The beta weights have a closed form solution if we have two subspaces, so the optimal values given the data, centers and V can be computed.
    See supplement of Lukas Miklautz, Lena G. M. Bauer, Dominik Mautz, Sebastian Tschiatschek, Christian Boehm, Claudia Plant: Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces. IJCAI 2021: 2826-2832
    here: https://gitlab.cs.univie.ac.at/lukas/acedec_public/-/blob/master/supplement.pdf

    Parameters
    ----------
    data : torch.Tensor
        input data
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    V : torch.Tensor
        orthogonal rotation matrix
    batch_size : int
        size of the data batches (default: 32)

    Returns
    -------
    optimal_beta_weights: torch.Tensor
        a c x d vector containing the optimal weights for the softmax to indicate which dimensions d are important for each clustering c.
    """
    dataloader = get_dataloader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    device = V.device
    with torch.no_grad():
        # calculate kmeans losses for each clustering
        km_losses = [[] for _ in centers]
        for batch in dataloader:
            batch = batch[1].to(device)
            z_rot = torch.matmul(batch, V)
            for i, centers_i in enumerate(centers):
                centers_i = centers_i.to(device)
                weighted_squared_diff = squared_euclidean_distance(z_rot.unsqueeze(1), centers_i.unsqueeze(1))
                assignments = weighted_squared_diff.detach().sum(2).argmin(1)
                if len(set(assignments.tolist())) > 1:
                    one_hot_mask = int_to_one_hot(assignments, centers_i.shape[0])
                    weighted_squared_diff_masked = weighted_squared_diff * one_hot_mask.unsqueeze(2)
                else:
                    weighted_squared_diff_masked = weighted_squared_diff

                km_losses[i].append(weighted_squared_diff_masked.detach().cpu())
                centers_i = centers_i.cpu()
        for i, km_loss in enumerate(km_losses):
            # Sum over samples and centers
            km_losses[i] = torch.cat(km_loss, 0).sum(0).sum(0)
        # calculate beta_weights for each dimension and clustering based on kmeans losses
        best_weights = []
        best_weights.append(optimal_beta(km_losses[0], km_losses[1]))
        best_weights.append(optimal_beta(km_losses[1], km_losses[0]))
        best_weights = torch.stack(best_weights)
    return best_weights


def beta_weights_init(P: list, n_dims: int, high_value: float = 0.9) -> torch.Tensor:
    """
    Initializes parameters of the softmax such that betas will be set to high_value in dimensions which form a cluster subspace according to P
    and set to (1 - high_value)/(len(P) - 1) for the other clusterings.

    Parameters
    ----------
    P : list
        list containing projections for each subspace
    n_dims : int
        dimensionality of the embedded data
    high_value : float
        value that should be initially used to indicate strength of assignment of a specific dimension to the clustering (default: 0.9)

    Returns
    ----------
    beta_weights : torch.Tensor
        initialized weights that are input in the softmax to get the betas.
    """
    weight_high = 1.0
    n_sub_clusterings = len(P)
    beta_hard = np.zeros((n_sub_clusterings, n_dims), dtype=np.float32)
    for sub_i, p in enumerate(P):
        for dim in p:
            beta_hard[sub_i, dim] = 1.0
    low_value = 1.0 - high_value
    weight_high_exp = np.exp(weight_high)
    # Because high_value = weight_high/(weight_high +low_classes*weight_low)
    n_low_classes = len(P) - 1
    weight_low_exp = weight_high_exp * (1.0 - high_value) / (high_value * n_low_classes)
    weight_low = np.log(weight_low_exp)
    beta_soft_weights = beta_hard * (weight_high - weight_low) + weight_low
    return torch.tensor(beta_soft_weights, dtype=torch.float32)


def calculate_beta_weight(data: torch.Tensor, centers: list, V: torch.Tensor, P: list,
                          high_beta_value: float = 0.9) -> torch.Tensor:
    """
    The beta weights have a closed form solution if we have two subspaces, so the optimal values given the data, centers and V can be computed.
    See supplement of Lukas Miklautz, Lena G. M. Bauer, Dominik Mautz, Sebastian Tschiatschek, Christian Boehm, Claudia Plant: Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces. IJCAI 2021: 2826-2832
    here: https://gitlab.cs.univie.ac.at/lukas/acedec_public/-/blob/master/supplement.pdf
    For number of subspaces > 2, we calculate the beta weight assuming that an assigned subspace should have a weight of 0.9.

    Parameters
    ----------
    data : torch.Tensor
        input data
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    V : torch.Tensor
        orthogonal rotation matrix
    P : list
        list containing projections for each subspace
    high_beta_value : float
        value that should be initially used to indicate strength of assignment of a specific dimension to the clustering (default: 0.9)

    Returns
    -------
    beta_weights: torch.Tensor
        a c x d vector containing the weights for the softmax to indicate which dimensions d are important for each clustering c.

    Raises
    -------
    ValueError: If number of clusterings is smaller than 2
    """
    n_clusterings = len(centers)
    if n_clusterings == 2:
        beta_weights = calculate_optimal_beta_weights_special_case(data=data, centers=centers, V=V)
    elif n_clusterings > 2:
        beta_weights = beta_weights_init(P=P, n_dims=data.shape[1], high_value=high_beta_value)
    else:
        raise ValueError(f"Number of clusterings is {n_clusterings}, but should be >= 2")
    return beta_weights
