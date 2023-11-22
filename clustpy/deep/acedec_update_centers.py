import torch



# TODO: implement labels
def acedec_update_center(self, data, one_hot_mask, subspace_id, labels, epoch_i):
    """Inplace update of centers of a clusterings in subspace=subspace_id in a mini-batch fashion.

    Parameters
    ----------
    data : torch.tensor, data points, can also be a mini-batch of points
    one_hot_mask : torch.tensor, one hot encoded matrix of cluster assignments
    subspace_id : int, integer which indicates which subspace the cluster to be updated are in

    Raises
    ----------
    ValueError: If None values are encountered.
    """

    if self.centers[subspace_id].shape[0] == 1:
        # Shared space update with only one cluster
        self.centers[subspace_id] = self.centers[subspace_id] * 0.5 + data.mean(0).unsqueeze(0) * 0.5
    else:
        batch_cluster_sums = (data.unsqueeze(1) * one_hot_mask.unsqueeze(2)).sum(0)
        mask_sum = one_hot_mask.sum(0).unsqueeze(1)
        if (mask_sum == 0).sum().int().item() != 0:
            idx = (mask_sum == 0).nonzero()[:, 0].detach().cpu()
            self.lonely_centers_count[subspace_id][idx] += 1

        # In case mask sum is zero batch cluster sum is also zero so we can add a small constant to mask sum and center_lr
        # Avoid division by a small number
        mask_sum += 1e-8
        # Use weighted average
        nonzero_mask = (mask_sum.squeeze(1) != 0)
        self.mask_sum[subspace_id][nonzero_mask] = self.center_lr * mask_sum[nonzero_mask] + (1 - self.center_lr) * \
                                                   self.mask_sum[subspace_id][nonzero_mask]

        per_center_lr = 1.0 / (1 + self.mask_sum[subspace_id][nonzero_mask])
        self.centers[subspace_id] = (1.0 - per_center_lr) * self.centers[subspace_id][
            nonzero_mask] + per_center_lr * batch_cluster_sums[nonzero_mask] / mask_sum[nonzero_mask]
        if torch.isnan(self.centers[subspace_id]).sum() > 0:
            raise ValueError(
                f"Found nan values\n self.centers[subspace_id]: {self.centers[subspace_id]}\n per_center_lr: {per_center_lr}\n self.mask_sum[subspace_id]: {self.mask_sum[subspace_id]}\n ")