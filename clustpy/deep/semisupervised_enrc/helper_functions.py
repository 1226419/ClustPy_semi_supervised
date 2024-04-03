import torch

import numpy as np
from clustpy.deep._utils import  squared_euclidean_distance
from sklearn.metrics import normalized_mutual_info_score


"""
===================== Helper Functions =====================
"""


class _IdentityAutoencoder(torch.nn.Module):
    """
    Convenience class to avoid reimplementation of the remaining ENRC pipeline for the initialization.
    Encoder and decoder are here just identity functions implemented via lambda x:x.

    Attributes
    ----------
    encoder : function
        the encoder part
    decoder : function
        the decoder part
    """

    def __init__(self):
        super(_IdentityAutoencoder, self).__init__()

        self.encoder = lambda x: x
        self.decoder = lambda x: x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the encoder function to x.

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of points

        Returns
        -------
        encoded : torch.Tensor
            the encoeded data point
        """
        return self.encoder(x)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Apply the decoder function to embedded.

        Parameters
        ----------
        embedded : torch.Tensor
            embedded data point, can also be a mini-batch of embedded points

        Returns
        -------
        decoded : torch.Tensor
            returns the reconstruction of embedded
        """
        return self.decoder(embedded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies both the encode and decode function.
        The forward function is automatically called if we call self(x).

        Parameters
        ----------
        x : torch.Tensor
            input data point, can also be a mini-batch of embedded points

        Returns
        -------
        reconstruction : torch.Tensor
            returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction


def _get_P(betas: torch.Tensor, centers: list, shared_space_variation: float = 0.05) -> float:
    """
    Converts the softmax betas back to hard assignments P and returns them as a list.

    Parameters
    ----------
    betas : torch.Tensor
        c x d soft assignment weights matrix for c clusterings and d dimensions.
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    shared_space_variation : float
        specifies how much beta in the shared space is allowed to diverge from the uniform distribution. Only needed if a shared space (space with one cluster) exists (default: 0.05)

    Returns
    ----------
    P : list
        list containing indices for projections for each clustering
    """
    # Check if a shared space with a single cluster center exist
    shared_space_idx = [i for i, centers_i in enumerate(centers) if centers_i.shape[0] == 1]
    if shared_space_idx:
        # Specifies how much beta in the shared space is allowed to diverge from the uniform distribution
        shared_space_idx = shared_space_idx[0]
        equal_threshold = 1.0 / betas.shape[0]
        # Increase Weight of shared space dimensions that are close to the uniform distribution
        equal_threshold -= shared_space_variation
        betas[shared_space_idx][betas[shared_space_idx] > equal_threshold] += 1

    # Select highest assigned dimensions to P
    max_assigned_dims = betas.argmax(0)
    P = [[] for _ in range(betas.shape[0])]
    for dim_i, cluster_subspace_id in enumerate(max_assigned_dims):
        P[cluster_subspace_id].append(dim_i)
    return P


def _rotate(z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Rotate the embedded data ponint z using the orthogonal rotation matrix V.

    Parameters
    ----------
    V : torch.Tensor
        orthogonal rotation matrix
    z : torch.Tensor
        embedded data point, can also be a mini-batch of points

    Returns
    -------
    z_rot : torch.Tensor
        the rotated embedded data point
    """
    return torch.matmul(z, V)


def _rotate_back(z_rot: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Rotate a rotated embedded data point back to its original state.

    Parameters
    ----------
    z_rot : torch.Tensor
        rotated and embedded data point, can also be a mini-batch of points
    V : torch.Tensor
        orthogonal rotation matrix

    Returns
    -------
    z : torch.Tensor
        the back-rotated embedded data point
    """
    return torch.matmul(z_rot, V.t())


def enrc_predict(z: torch.Tensor, V: torch.Tensor, centers: list, subspace_betas: torch.Tensor,
                 use_P: bool = False) -> np.ndarray:
    """
    Predicts the labels for each clustering of an input z.

    Parameters
    ----------
    z : torch.Tensor
        embedded input data point, can also be a mini-batch of embedded points
    V : torch.tensor
        orthogonal rotation matrix
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    subspace_betas : torch.Tensor
        weights for each dimension per clustering. Calculated via softmax(beta_weights).
    use_P: bool
        if True then P will be used to hard select the dimensions for each clustering, else the soft subspace_beta weights are used (default: False)

    Returns
    -------
    predicted_labels : np.ndarray
        n x c matrix, where n is the number of data points in z and c is the number of clusterings.
    """
    z_rot = _rotate(z, V)
    if use_P:
        P = _get_P(betas=subspace_betas.detach(), centers=centers)
    labels = []
    for i, centers_i in enumerate(centers):
        if use_P:
            weighted_squared_diff = squared_euclidean_distance(z_rot[:, P[i]], centers_i[:, P[i]])
        else:
            weighted_squared_diff = squared_euclidean_distance(z_rot, centers_i, weights=subspace_betas[i, :])
        labels_sub = weighted_squared_diff.argmin(1)
        labels_sub = labels_sub.detach().cpu().numpy().astype(np.int32)
        labels.append(labels_sub)
    return np.stack(labels).transpose()


def enrc_predict_batchwise(V: torch.Tensor, centers: list, subspace_betas: torch.Tensor, model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader, device: torch.device = torch.device("cpu"),
                           use_P: bool = False) -> np.ndarray:
    """
    Predicts the labels for each clustering of a dataloader in a mini-batch manner.

    Parameters
    ----------
    V : torch.Tensor
        orthogonal rotation matrix
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    subspace_betas : torch.Tensor
        weights for each dimension per clustering. Calculated via softmax(beta_weights).
    model : torch.nn.Module
        the input model for encoding the data
    dataloader : torch.utils.data.DataLoader
        dataloader to be used for prediction
    device : torch.device
        device to be predicted on (default: torch.device('cpu'))
    use_P: bool
        if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used (default: False)

    Returns
    -------
    predicted_labels : np.ndarray
        n x c matrix, where n is the number of data points in z and c is the number of clusterings.
    """
    model.eval()
    predictions = []
    print("enrc predict batchwise centers", centers)
    print("enrc predict batchwise subspace_betas", subspace_betas)
    print("enrc predict batchwise V", V)
    print("enrc predict batchwise use_P", use_P)
    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[1].to(device)
            z = model.encode(batch_data)
            pred_i = enrc_predict(z=z, V=V, centers=centers, subspace_betas=subspace_betas, use_P=use_P)
            predictions.append(pred_i)
    print("enrc predict batchwise", np.concatenate(predictions))
    print("enrc predict batchwise shape", np.concatenate(predictions).shape)
    return np.concatenate(predictions)


def enrc_encode_decode_batchwise_with_loss(V: torch.Tensor, centers: list, model: torch.nn.Module,
                                           dataloader: torch.utils.data.DataLoader,
                                           device: torch.device = torch.device("cpu"),
                                           loss_fn: torch.nn.modules.loss._Loss = None) -> np.ndarray:
    """
    Encode and Decode input data of a dataloader in a mini-batch manner with ENRC.

    Parameters
    ----------
    V : torch.Tensor
        orthogonal rotation matrix
    centers : list
        list of torch.Tensor, cluster centers for each clustering
    model : torch.nn.Module
        the input model for encoding the data
    dataloader : torch.utils.data.DataLoader
        dataloader to be used for prediction
    device : torch.device
        device to be predicted on (default: torch.device('cpu'))
    loss_fn : torch.nn.modules.loss._Loss
            specifies loss function to be used for calculating reconstruction error (default: None)
    Returns
    -------
    enrc_encoding : np.ndarray
        n x d matrix, where n is the number of data points and d is the number of dimensions of z.
    enrc_decoding : np.ndarray
        n x D matrix, where n is the number of data points and D is the data dimensionality.
    reconstruction_error : flaot
        reconstruction error (will be None if loss_fn is not specified)
    """
    model.eval()
    reconstructions = []
    embeddings = []
    if loss_fn is None:
        loss = None
    else:
        loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[1].to(device)
            z = model.encode(batch_data)
            z_rot = _rotate(z=z, V=V)
            embeddings.append(z_rot.detach().cpu())
            z_rot_back = _rotate_back(z_rot=z_rot, V=V)
            reconstruction = model.decode(z_rot_back)
            if loss_fn is not None:
                loss += loss_fn(reconstruction, batch_data).item()
            reconstructions.append(reconstruction.detach().cpu())
    if loss_fn is not None:
        loss /= len(dataloader)
    embeddings = torch.cat(embeddings).numpy()
    reconstructions = torch.cat(reconstructions).numpy()
    return embeddings, reconstructions, loss


def _are_labels_equal(labels_new: np.ndarray, labels_old: np.ndarray, threshold: float = None) -> bool:
    """
    Check if the old labels and new labels are equal. Therefore check the nmi for each subspace_nr. If all are 1, labels
    have not changed.

    Parameters
    ----------
    labels_new: np.ndarray
        new labels list
    labels_old: np.ndarray
        old labels list
    threshold: float
        specifies how close the two labelings should match (default: None)

    Returns
    ----------
    changed : bool
        True if labels for all subspaces are the same
    """
    if labels_new is None or labels_old is None or labels_new.shape[1] != labels_old.shape[1]:
        return False

    if threshold is None:
        v = 1
    else:
        v = 1 - threshold
    return all(
        [normalized_mutual_info_score(labels_new[:, i], labels_old[:, i], average_method="arithmetic") >= v for i in
         range(labels_new.shape[1])])
