from __future__ import annotations
import torch
import numpy as np
from clustpy.deep._utils import int_to_one_hot, squared_euclidean_distance

from typing import TYPE_CHECKING

"""
===================== Cluster Reinitialization Strategy =====================
"""
if TYPE_CHECKING:
    from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_module import _ENRC_Module


def _calculate_rotated_embeddings_and_distances_for_n_samples(enrc: _ENRC_Module, model: torch.nn.Module,
                                                              dataloader: torch.utils.data.DataLoader, n_samples: int,
                                                              center_id: int, subspace_id: int, device: torch.device,
                                                              calc_distances: bool = True) -> (
        torch.Tensor, torch.Tensor):
    """
    Helper function for calculating the distances and embeddings for n_samples in a mini-batch fashion.

    Parameters
    ----------
    enrc : _ENRC_Module
        The ENRC Module
    model : torch.nn.Module
        The autoencoder
    dataloader : torch.utils.data.DataLoader
        dataloader from which data is randomly sampled
    n_samples : int
        the number of samples
    center_id : int
        the id of the center
    subspace_id : int
        the id of the subspace
    device : torch.device
        device to be trained on
    calc_distances : bool
        specifies if the distances between all not lonely centers to embedded data points should be calculated

    Returns
    -------
    tuple : (torch.Tensor, torch.Tensor)
        the rotated embedded data points
        the distances (if calc_distancesis True)
    """
    changed = True
    sample_count = 0
    subspace_betas = enrc.subspace_betas()[subspace_id, :]
    subspace_centers = enrc.centers[subspace_id]
    embedding_rot = []
    dists = []
    for batch in dataloader:
        batch = batch[1].to(device)
        if (batch.shape[0] + sample_count) > n_samples:
            # Remove samples from the batch that are too many.
            # Assumes that dataloader randomly samples minibatches,
            # so removing the last objects does not matter
            diff = (batch.shape[0] + sample_count) - n_samples
            batch = batch[:-diff]
        z_rot = enrc.rotate(model.encode(batch))
        embedding_rot.append(z_rot.detach().cpu())

        if calc_distances:
            # Calculate distance from all not lonely centers to embedded data points
            idx_other_centers = [i for i in range(subspace_centers.shape[0]) if i != center_id]
            weighted_squared_diff = squared_euclidean_distance(z_rot, subspace_centers[idx_other_centers],
                                                               weights=subspace_betas)
            dists.append(weighted_squared_diff.detach().cpu())

        # Increase sample_count by batch size.
        sample_count += batch.shape[0]
        if sample_count >= n_samples:
            break
    embedding_rot = torch.cat(embedding_rot, 0)
    if calc_distances:
        dists = torch.cat(dists, 0)
    else:
        dists = None
    return embedding_rot, dists


def _split_most_expensive_cluster(distances: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Splits most expensive cluster calculated based on the k-means loss and returns a new centroid.
    The new centroid is the one which is worst represented (largest distance) by the most expensive cluster centroid.

    Parameters
    ----------
    distances : torch.Tensor
        n x n distance matrix, where n is the size of the mini-batch.
    z : torch.Tensor
        n x d embedded data point, where n is the size of the mini-batch and d is the dimensionality of the embedded space.

    Returns
    -------
    center : torch.Tensor
        new center
    """
    kmeans_loss = distances.sum(0)
    costly_centroid_idx = kmeans_loss.argmax()
    max_idx = distances[costly_centroid_idx, :].argmax()
    return z[max_idx]


def _random_reinit_cluster(embedded: torch.Tensor) -> torch.Tensor:
    """
    Reinitialize random cluster centers.

    Parameters
    ----------
    embedded : torch.Tensor
        The embedded data points

    Returns
    -------
    center : torch.Tensor
        The random center
    """
    rand_indices = np.random.randint(low=0, high=embedded.shape[0], size=1)
    random_perturbation = torch.empty_like(embedded[rand_indices]).normal_(mean=embedded.mean().item(),
                                                                           std=embedded.std().item())
    center = embedded[rand_indices] + 0.0001 * random_perturbation
    return center


def reinit_centers(enrc: _ENRC_Module, subspace_id: int, dataloader: torch.utils.data.DataLoader,
                   model: torch.nn.Module,
                   n_samples: int = 512, kmeans_steps: int = 10, split: str = "random", debug: bool = False) -> None:
    """
    Reinitializes centers that have been lost, i.e. if they did not get any data point assigned. Before a center is reinitialized,
    this method checks whether a center has not get any points assigned over several mini-batch iterations and if this count is higher than
    enrc.reinit_threshold the center will be reinitialized.

    Parameters
    ----------
    enrc : _ENRC_Module
        torch.nn.Module instance for the ENRC algorithm
    subspace_id : int
        integer which indicates which subspace the cluster to be checked are in.
    dataloader : torch.utils.data.DataLoader
        dataloader from which data is randomly sampled. Important shuffle=True needs to be set, because n_samples random samples are drawn.
    model : torch.nn.Module
        autoencoder model used for the embedding
    n_samples : int
        number of samples that should be used for the reclustering (default: 512)
    kmeans_steps : int
        number of mini-batch kmeans steps that should be conducted with the new centroid (default: 10)
    split : str
        {'random', 'cost'}, default='random', select how clusters should be split for renitialization.
        'random' : split a random point from the random sample of size=n_samples.
        'cost' : split the cluster with max kmeans cost.
    debug : bool
        if True than training errors will be printed (default: True)
    """
    N = len(dataloader.dataset)
    if n_samples > N:
        if debug: print \
            (f"WARNING: n_samples={n_samples} > number of data points={N}. Set n_samples=number of data points")
        n_samples = N
    # Assumes that enrc and model are on the same device
    device = enrc.V.device
    with torch.no_grad():
        k = enrc.centers[subspace_id].shape[0]
        subspace_betas = enrc.subspace_betas()
        for center_id, count_i in enumerate(enrc.lonely_centers_count[subspace_id].flatten()):
            if count_i > enrc.reinit_threshold:
                if debug: print(f"Reinitialize cluster {center_id} in subspace {subspace_id}")
                if split == "cost":
                    embedding_rot, dists = _calculate_rotated_embeddings_and_distances_for_n_samples(enrc, model,
                                                                                                     dataloader,
                                                                                                     n_samples,
                                                                                                     center_id,
                                                                                                     subspace_id,
                                                                                                     device)
                    new_center = _split_most_expensive_cluster(distances=dists, z=embedding_rot)
                elif split == "random":
                    embedding_rot, _ = _calculate_rotated_embeddings_and_distances_for_n_samples(enrc, model,
                                                                                                 dataloader, n_samples,
                                                                                                 center_id, subspace_id,
                                                                                                 device,
                                                                                                 calc_distances=False)
                    new_center = _random_reinit_cluster(embedding_rot)
                else:
                    raise NotImplementedError(f"split={split} is not implemented. Has to be 'cost' or 'random'.")
                enrc.centers[subspace_id][center_id, :] = new_center.to(device)

                embeddingloader = torch.utils.data.DataLoader(embedding_rot, batch_size=dataloader.batch_size,
                                                              shuffle=False, drop_last=False)
                # perform mini-batch kmeans steps
                batch_cluster_sums = 0
                mask_sum = 0
                for step_i in range(kmeans_steps):
                    for z_rot in embeddingloader:
                        z_rot = z_rot.to(device)
                        weighted_squared_diff = squared_euclidean_distance(z_rot, enrc.centers[subspace_id],
                                                                           weights=subspace_betas[subspace_id, :])
                        assignments = weighted_squared_diff.detach().argmin(1)
                        one_hot_mask = int_to_one_hot(assignments, k)
                        batch_cluster_sums += (z_rot.unsqueeze(1) * one_hot_mask.unsqueeze(2)).sum(0)
                        mask_sum += one_hot_mask.sum(0)
                    nonzero_mask = (mask_sum != 0)
                    enrc.centers[subspace_id][nonzero_mask] = batch_cluster_sums[nonzero_mask] / mask_sum[
                        nonzero_mask].unsqueeze(1)
                    # Reset mask_sum
                    enrc.mask_sum[subspace_id] = mask_sum.unsqueeze(1)
                # lonely_centers_count is reset
                enrc.lonely_centers_count[subspace_id][center_id] = 0
