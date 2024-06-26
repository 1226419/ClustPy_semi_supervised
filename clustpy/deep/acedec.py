import torch
import numpy as np
from typing import Union
from clustpy.deep._utils import int_to_one_hot, squared_euclidean_distance, encode_batchwise, detect_device, \
    set_torch_seed
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._train_utils import get_trained_autoencoder
from clustpy.utils.plots import plot_scatter_matrix, plot_2d_data
from clustpy.deep.enrc import enrc_init, enrc_predict_batchwise,_get_P, _rotate, _rotate_back, enrc_predict, \
    reinit_centers, _are_labels_equal, available_init_strategies, calculate_optimal_beta_weights_special_case, \
    enrc_encode_decode_batchwise_with_loss
from clustpy.deep.acedec_init import acedec_init
from clustpy.deep.acedec_recluster import acedec_recluster
from clustpy.deep.dcn import DCN
from sklearn.base import BaseEstimator, ClusterMixin
from clustpy.deep.acedec_predict import acedec_predict, acedec_predict_batchwise
from clustpy.deep.dec import _dec_predict, _dec_compression_loss_fn
from sklearn.utils.validation import check_random_state
"""
===================== ACeDeC - MODULE =====================
"""
class _ACeDeC_Module(torch.nn.Module):
    """Create an instance of the ACeDeC torch.nn.Module.

    Parameters
    ----------
    centers : list, list containing the cluster centers for each clustering
    V : numpy.ndarray, orthogonal rotation matrix
    P : list,  list containing projections for each clustering
    beta_init_value : float, default=0.9, initial values of beta weights. Is ignored if beta_weights is not None.
    degree_of_space_distortion : float, default=1.0, weight of the cluster loss term. The higher it is set the more the
     embedded space will be shaped to the assumed cluster structure.
    degree_of_space_preservation : float, default=1.0, weight of regularization loss term, e.g., reconstruction loss.
    center_lr : float, default=0.5, weight for updating the centers via mini-batch k-means. Has to be set between 0 and
        1. If set to 1.0 than only the mini-batch centroid will be used,
              neglecting the past state and if set to 0 then no update is happening.
    rotate_centers : bool, default=False, if True then centers are multiplied with V before they are used, because ENRC assumes that the centers lie already in the V-rotated space.
    beta_weights : np.ndarray, default=None, initial beta weights for the softmax (optional). If not None, then beta_init_value will be ignored.
    Attributes
    ----------
    lonely_centers_count : list of np.ndarrays, count indicating how often a center in a clustering has not received any updates, because no points were assigned to it.
                           The lonely_centers_count of a center is reset if it has been reinitialized.
    mask_sum : list of torch.tensors, contains the average number of points assigned to each cluster in each clustering over the training.
    reinit_threshold : int, threshold that indicates when a cluster should be reinitialized. Starts with 1 and increases during training with int(np.sqrt(i+1)), where i is the number of mini-batch iterations.

    Raises
    ----------
    ValueError : if center_lr is not in [0,1]

    References
    ----------
    Miklautz, Lukas and Bauer, Lena G. M. and Mautz, Dominik and Tschiatschek, Sebastian and Böhm, Christian and
    Plant, Claudia.  "Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces."
    Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21), 2826--2832, 19.-27.08.2021.
    """

    def __init__(self, centers, P, V, beta_init_value=0.9, degree_of_space_distortion=1.0,
                 degree_of_space_preservation=1.0, center_lr=0.5, rotate_centers=False, beta_weights=None,
                 cluster_space_module=DCN, noise_space_module=DCN, cluster_assignment="acedec",
                 cluster_assignment_kwargs=None, reclustering="like_init", reclustering_kwargs=None,
                 update_cluster_centers="acedec", update_cluster_centers_kwargs=None,
                 loss_calculation="acedec", loss_calculation_kwargs=None, augmentation_invariance: bool = False,
                 ):
        super().__init__()
        self.P = P
        self.m = [len(P_i) for P_i in self.P]
        if beta_weights is None:
            beta_weights = beta_weights_init(self.P, n_dims=centers[0].shape[1], high_value=beta_init_value)
        else:
            beta_weights = torch.tensor(beta_weights).float()
        self.beta_weights = torch.nn.Parameter(beta_weights, requires_grad=True)
        self.V = torch.nn.Parameter(torch.tensor(V, dtype=torch.float), requires_grad=True)
        if rotate_centers:
            centers = [np.matmul(centers_sub, V) for centers_sub in centers]
        if update_cluster_centers == "acedec":
            self.centers = [torch.tensor(centers_sub, dtype=torch.float32) for centers_sub in centers]
        elif update_cluster_centers == "dec":
            #TODO both centers learnable ? yes
            self.centers = torch.nn.Parameter(torch.tensor(centers), requires_grad=True)
        if not (0 <= center_lr <= 1):
            raise ValueError(f"center_lr={center_lr}, but has to be in [0,1].")
        self.center_lr = center_lr
        self.lonely_centers_count = []
        self.mask_sum = []
        self.augmentation_invariance = augmentation_invariance
        for centers_i in self.centers:
            self.lonely_centers_count.append(np.zeros((centers_i.shape[0], 1)).astype(int))
            self.mask_sum.append(torch.zeros((centers_i.shape[0], 1)))
        self.reinit_threshold = 1
        self.cluster_assignment = cluster_assignment
        self.cluster_assignment_kwargs = cluster_assignment_kwargs
        self.reclustering = reclustering
        self.reclustering_kwargs = reclustering_kwargs
        self.update_cluster_centers = update_cluster_centers
        self.update_cluster_centers_kwargs = update_cluster_centers_kwargs
        self.loss_calculation = loss_calculation
        self.loss_calculation_kwargs = loss_calculation_kwargs
        # self.cluster_space_module = cluster_space_module(centers)
        # self.noise_space_module = noise_space_module(centers)
        self.degree_of_space_distortion = degree_of_space_distortion
        self.degree_of_space_preservation = degree_of_space_preservation

    def to_device(self, device):
        """Loads all ACeDeC parameters to device that are needed during the training and prediction (including the
        learnable parameters).
        This function is preferred over the to(device) function.
        """
        self.to(device)
        self.centers = [c_i.to(device) for c_i in self.centers]
        self.mask_sum = [i.to(device) for i in self.mask_sum]
        return self

    def subspace_betas(self):
        """Returns a len(P) x d matrix with softmax weights, where d is the number of dimensions of the embedded space, indicating
           which dimensions belongs to which clustering.
        """
        dimension_assignments = torch.nn.functional.softmax(self.beta_weights, dim=0)
        return dimension_assignments

    def get_P(self):
        """
        Converts the soft beta weights back to hard assignments P and returns them as a list.

        Returns
        -------
        P : list
            list containing indices for projections for each clustering
        """
        P = _get_P(betas=self.subspace_betas().detach().cpu(), centers=self.centers)
        return P

    def rotate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Rotate the embedded data ponint z using the orthogonal rotation matrix V.

        Parameters
        ----------
        z : torch.Tensor
            embedded data point, can also be a mini-batch of points

        Returns
        -------
        z_rot : torch.Tensor
            the rotated embedded data point
        """
        z_rot = _rotate(z, self.V)
        return z_rot

    def rotate_back(self, z_rot: torch.Tensor) -> torch.Tensor:
        """
        Rotate a rotated embedded data point back to its original state.

        Parameters
        ----------
        z_rot : torch.Tensor
            rotated and embedded data point, can also be a mini-batch of points

        Returns
        -------
        z : torch.Tensor
            the back-rotated embedded data point
        """
        z = _rotate_back(z_rot, self.V)
        return z

    def rotation_loss(self) -> torch.Tensor:
        """
        Computes how much the rotation matrix self.V diverges from an orthogonal matrix by calculating |V^T V - I|.
        For an orthogonal matrix this difference is 0, as V^T V=I.

        Returns
        -------
        rotation_loss : torch.Tensor
            the average absolute difference between V^T times V - the identity matrix I.
        """
        ident = torch.matmul(self.V.t(), self.V).detach().cpu()
        rotation_loss = (ident - torch.eye(n=ident.shape[0])).abs().mean()
        return rotation_loss

    def update_center(self, data, one_hot_mask, subspace_id, labels, epoch_i):
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


    def update_centers(self, z_rot, assignment_matrix_dict, labels=None, epoch_i=None):
        """Inplace update of all centers in all clusterings in a mini-batch fashion.

        Parameters
        ----------
        z_rot : torch.tensor, rotated data point, can also be a mini-batch of points
        assignment_matrix_dict : dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments
        labels : torch.tensor, labels of the data points, can also be a mini-batch of points
        epoch_i : int, current epoch
        """
        if self.update_cluster_centers == "acedec":
            for subspace_i in range(len(self.centers)):
                self.update_center(z_rot.detach(),
                                assignment_matrix_dict[subspace_i],
                                subspace_id=subspace_i, labels=labels, epoch_i=epoch_i)
        elif self.update_cluster_centers == "dec":
            print("No update step needed - as centers are learned")
        else:
            raise ValueError(f"init={self.update_cluster_centers} is not implemented.")

    def forward(self, z: torch.Tensor, assignment_matrix_dict: dict = None, batch_labels: torch.Tensor = None,
                epoch_i: int = None):
        """Calculates the k-means loss and cluster assignments for each clustering.

        Parameters
        ----------
        z : torch.Tensor
            embedded input data point, can also be a mini-batch of embedded points
        assignment_matrix_dict : dict
            dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments (default: None)
        batch_labels : torch.tensor
            labels of the input data points, can also be a mini-batch of labels
        epoch_i : int
            current epoch

        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor, dict)
            subspace_losses : averaged sum of all k-means losses for each clustering
            z_rot : the rotated embedded point
            z_rot_back : the back rotated embedded point
            assignment_matrix_dict : dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments
        """
        z_rot = self.rotate(z)
        z_rot_back = self.rotate_back(z_rot)

        subspace_betas = self.subspace_betas()
        subspace_losses = 0

        if assignment_matrix_dict is None:
            assignment_matrix_dict = {}
            overwrite_assignments = True
        else:
            overwrite_assignments = False
        """
        for i, centers_i in enumerate(self.centers):
            # handle the cluster spaces - one cluster space for each label
            if i < len(self.centers) - 1:
                assert len(centers_i) > 1, "Each cluster space should have more than one cluster"
                # getting the distances to the cluster centers and weigh them by the betas
                weighted_squared_diff = squared_euclidean_distance(tensor1=z_rot, tensor2=centers_i.detach(),
                                                                   weights=subspace_betas[i, :])
                weighted_squared_diff /= z_rot.shape[0]
                assignments = weighted_squared_diff.detach().argmin(1)

                current_labels = batch_labels
                if current_labels.ndim > 1:
                    current_labels = batch_labels[:, i]
                # get a mask that is 1 if the current_label value is -1 and 0 otherwise
                current_labels_mask = (current_labels != -1).int()
                # replace assignments values with current_labels values where current_labels_mask is 1
                # that means if we have a label we use the label assignment instead of the closest cluster assignment
                assignments = assignments * (1 - current_labels_mask) + current_labels * current_labels_mask
                one_hot_mask = int_to_one_hot(assignments, centers_i.shape[0])
                weighted_squared_diff_masked = weighted_squared_diff * one_hot_mask
                subspace_losses += weighted_squared_diff_masked.sum()
                assignment_matrix_dict[i] = one_hot_mask
            # Handle the noise subspace
            else:
                assert len(centers_i) == 1, "Noise subspace should only have one cluster"

                weighted_squared_diff = squared_euclidean_distance(tensor1=z_rot, tensor2=centers_i.detach(),
                                                                   weights=subspace_betas[i, :])
                weighted_squared_diff /= z_rot.shape[0]
                subspace_losses += weighted_squared_diff.sum()
                one_hot_mask = torch.ones([weighted_squared_diff.shape[0], 1], dtype=torch.float,
                                          device=weighted_squared_diff.device)
                assignment_matrix_dict[i] = one_hot_mask
        """
        for i, centers_i in enumerate(self.centers):
            weighted_squared_diff = squared_euclidean_distance(z_rot, centers_i.detach(), weights=subspace_betas[i, :])
            weighted_squared_diff /= z_rot.shape[0]

            if overwrite_assignments:
                assignments = weighted_squared_diff.detach().argmin(1)
                one_hot_mask = int_to_one_hot(assignments, centers_i.shape[0])
                assignment_matrix_dict[i] = one_hot_mask
            else:
                one_hot_mask = assignment_matrix_dict[i]
            weighted_squared_diff_masked = weighted_squared_diff * one_hot_mask
            subspace_losses += weighted_squared_diff_masked.sum()

        subspace_losses = subspace_losses / subspace_betas.shape[0]
        return subspace_losses, z_rot, z_rot_back, assignment_matrix_dict

    def predict(self, z, use_P=False):
        """Predicts the labels for each clustering of an input z.

        Parameters
        ----------
        z : torch.tensor, embedded input data point, can also be a mini-batch of embedded points
        use_P: bool, default=False, if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used

        Returns
        -------
        predicted_labels : n x c matrix, where n is the number of data points in z and c is the number of clusterings.
        """
        return acedec_predict(z=z, V=self.V, centers=self.centers, subspace_betas=self.subspace_betas(), use_P=use_P)

    def predict_batchwise(self, model, dataloader, device=torch.device("cpu"), use_P=False):
        """Predicts the labels for each clustering of a dataloader in a mini-batch manner.

        Parameters
        ----------
        model : torch.nn.Module, the input model for encoding the data
        dataloader : torch.utils.data.DataLoader, dataloader to be used for prediction
        device : torch.device, default=torch.device('cpu'), device to be predicted on
        use_P: bool, default=False, if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used

        Returns
        -------
        predicted_labels : n x c matrix, where n is the number of data points in z and c is the number of clusterings.
        """
        return acedec_predict_batchwise(V=self.V, centers=self.centers, model=model, dataloader=dataloader,
                                      subspace_betas=self.subspace_betas(), device=device, use_P=use_P)


    def recluster(self, y, dataloader, model, device=torch.device('cpu'), rounds=1, optimizer_params: dict = {}, optimizer_class: torch.optim.Optimizer = None,):
        """Recluster ACeDeC inplace using NrKMeans or SGD (depending on the data set size, see init='auto' for details).
           Can lead to improved and more stable performance.
           Updates self.P, self.beta_weights, self.V and self.centers.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader, dataloader to be used for prediction
        model : torch.nn.Module, the input model for encoding the data
        device : torch.device, default=torch.device('cpu'), device to be predicted on
        rounds : int, default=1, number of repetitions of the reclustering procedure

        """

        # Extract parameters
        V = self.V.detach().cpu().numpy()
        n_clusters = [c.shape[0] for c in self.centers]

        # Encode data
        embedded_data = encode_batchwise(dataloader, model, device)
        embedded_rot = np.matmul(embedded_data, V)
        """
        print("embedded_rot", embedded_rot)
        
        # Apply reclustering in the rotated space, because V does not have to be orthogonal, so it could learn a mapping that is not recoverable by nrkmeans.
        centers_reclustered, P, new_V, beta_weights = enrc_init(data=embedded_rot, n_clusters=n_clusters, rounds=rounds,
                                                                max_iter=300, learning_rate=self.learning_rate,
                                                                init="auto", debug=False)
        print("Warning: labels not used for reclustering")  # TODO: add labels to reclustering
        """
        """
        centers_reclustered, P, new_V, beta_weights = acedec_recluster(y=y, embedded_data=embedded_rot,
                                                    n_clusters=n_clusters,
                                                    device=device, init="auto",
                                                    rounds=rounds, epochs=10, debug=False,
                                                    #input_centers=input_centers,
                                                    max_iter=300,
                                                    reclustering=self.reclustering,
                                                    reclustering_kwargs=self.reclustering_kwargs
                                                    )
        """
        # Apply reclustering in the rotated space, because V does not have to be orthogonal, so it could learn a mapping that is not recoverable by nrkmeans.
        print("reclustering values")
        print("n_clusters", n_clusters)
        print("rounds", rounds)
        print("optimizer_params", optimizer_params)
        print("optimizer_class", optimizer_class)
        print("reclustering_strategy", "acedec")
        print("init_kwargs", None)
        print("dataloader.batch_size", dataloader.batch_size)
        print("embedded_rot shape", embedded_rot.shape)
        centers_reclustered, P, new_V, beta_weights = enrc_init(data=embedded_rot, n_clusters=n_clusters, rounds=rounds,
                                                                max_iter=300, optimizer_params=optimizer_params,
                                                                optimizer_class=optimizer_class,
                                                                init="acedec", debug=False,
                                                                init_kwargs=None,
                                                                batch_size=dataloader.batch_size,
                                                                )
        # Update V, because we applied the reclustering in the rotated space
        new_V = np.matmul(V, new_V)

        # Assign reclustered parameters
        self.P = P
        self.m = [len(P_i) for P_i in self.P]
        self.beta_weights = torch.nn.Parameter(torch.from_numpy(beta_weights).float(), requires_grad=True)
        self.V = torch.nn.Parameter(torch.tensor(new_V, dtype=torch.float), requires_grad=True)
        self.centers = [torch.tensor(centers_sub, dtype=torch.float32) for centers_sub in centers_reclustered]
        self.to_device(device)

    #TODO: ground_truth_labels and old_predicted_labels and new_predicted_labels
    def fit(self, trainloader: torch.utils.data.DataLoader, evalloader: torch.utils.data.DataLoader,
            data, labels, optimizer, max_epochs, model, batch_size, loss_fn=torch.nn.MSELoss(),
            device=torch.device("cpu"), print_step=10, debug=True, scheduler=None, fix_rec_error=False,
            tolerance_threshold=None):
        """Trains ACEDEC and the autoencoder in place.

        Parameters
        ----------
        data : torch.tensor or np.ndarray, dataset to be used for training
        optimizer : torch.optim, parameterized optimizer to be used
        max_epochs : int, maximum number of epochs for training
        batch_size: int, batch size for dataloader
        loss_fn : torch.nn, default=torch.nn.MSELoss(), loss function to be used for reconstruction
        device : torch.device, default=torch.device('cpu'), device to be trained on
        print_step : int, default=10, specifies how often the losses are printed
        debug : bool, default=True, if True than training errors will be printed.
        scheduler : torch.optim.lr_scheduler, default=None, parameterized learning rate scheduler that should be used.
        fix_rec_error : bool, default=False, if set to True than reconstruction loss is weighted proportionally to the cluster loss. Only used for init='sgd'.
        tolerance_threshold : float, default=None, tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
                              for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
                              will train as long as the labels are not changing anymore.
        Returns
        -------
        model : torch.nn.Module, trained autoencoder
        enrc_module : torch.nn.Module, trained enrc module
        """
        """
        # Deactivate Batchnorm and dropout
        model.eval()
        model.to(device)
        self.to_device(device)

        # Save learning rate for reclustering
        self.learning_rate = optimizer.param_groups[0]["lr"]
        # Evalloader is used for checking label change. Only difference to the trainloader here is that shuffle=False.
        labels = labels[:, None] # what is that ??
        # data_tensor = torch.from_numpy(data)

        if trainloader is None and data is not None:
            trainloader = get_dataloader(data, additional_inputs=labels, batch_size=batch_size, shuffle=True, drop_last=True)
        elif trainloader is None and data is None:
            raise ValueError("trainloader and data cannot be both None.")
        if evalloader is None and data is not None:
            # Evalloader is used for checking label change. Only difference to the trainloader here is that shuffle=False.
            evalloader = get_dataloader(data, batch_size=batch_size, shuffle=False, drop_last=False)
        i = 0
        labels_old = None
        for epoch_i in range(max_epochs):
            for batch in trainloader:
                if len(batch) == 3: # maybe change this to use ndim and make sure order is always good
                    batch_labels = batch[2].to(device)
                    batch = batch[1].to(device)
                else:
                    batch = batch[1].to(device)

                embedded_data = model.encode(batch)
                if self.loss_calculation == "acedec":
                    subspace_loss, z_rot, z_rot_back, assignment_matrix_dict = self.forward(embedded_data, batch_labels,
                                                                                            epoch_i)
                    reconstruction = model.decode(z_rot_back)
                    rec_loss = loss_fn(reconstruction, batch)

                    # What is this for?
                    if fix_rec_error:
                        rec_weight = subspace_loss.item() / (rec_loss.item() + 1e-3)
                        if rec_weight < 1:
                            rec_weight = 1.0
                        rec_loss *= rec_weight
                    

                    summed_loss = subspace_loss + rec_loss
                    optimizer.zero_grad()
                    summed_loss.backward()
                    optimizer.step()

                    # Update Assignments and Centroids on GPU
                    with torch.no_grad():
                        self.update_centers(z_rot, assignment_matrix_dict, batch_labels, epoch_i) # TODO: update this
                        # function to use the new labels and epoch_i
                    # Check if clusters have to be reinitialized - noise cluster can't be reinitialized(TODO: check if this is true)
                    for subspace_i in range(len(self.centers)-1):
                        reinit_centers(enrc=self, subspace_id=subspace_i, dataloader=trainloader, model=model,
                                       n_samples=512, kmeans_steps=10)

                    # Increase reinit_threshold over time
                    self.reinit_threshold = int(np.sqrt(i + 1))

                    i += 1
                elif self.loss_calculation == "dec":

                    embedded_data_unlabeled = embedded_data[batch_labels[0, :] == -1]
                    embedded_data_labeled = embedded_data[batch_labels[0, :] != -1]
                    # Right place for optimal beta weights ?
                    subspace_optimal_beta_weights = calculate_optimal_beta_weights_special_case(embedded_data_unlabeled,
                                                                                                self.centers, self.V,
                                                                                                batch_size)
                    # TODO add reconstruction loss
                    # TODO handle reconstruction space
                    prediction = _dec_predict(self.centers, embedded_data_unlabeled, self.alpha,
                                              weights=subspace_optimal_beta_weights.detach())
                    compression_loss = _dec_compression_loss_fn(prediction)
                    prediction_labels = _dec_predict(self.centers, embedded_data_labeled, self.alpha,
                                                     weights=subspace_optimal_beta_weights.detach())
                    loss_labels = _dec_compression_loss_fn(prediction_labels)
                    summed_loss = compression_loss + loss_labels
                    optimizer.zero_grad()
                    summed_loss.backward()
                    optimizer.step()

            # TODO: what about rotation_loss ?
            if (epoch_i - 1) % print_step == 0 or epoch_i == (max_epochs - 1):
                with torch.no_grad():
                    # Rotation loss is calculated to check if its deviation from an orthogonal matrix
                    # rotation_loss = self.rotation_loss()
                    if debug:
                        print(
                            f"Epoch {epoch_i}/{max_epochs - 1}: summed_loss: {summed_loss.item():.4f}, "
                            f"subspace_losses: {subspace_loss.item():.4f}, rec_loss: {rec_loss.item():.4f}, "
                            f"rotation_loss: not calculated")
                        # plot the subspace and cluster centers
                       
                        #print("the type is")
                        #print(type(torch.from_numpy(data).float()))
                        #print(type(self.V))
                        #plot_2d_data(model.encode(torch.from_numpy(data).float())@self.V, labels, self.centers,
                        #             title=f"epoch_{epoch_i}")
                        


            if scheduler is not None:
                scheduler.step()

            # Check if labels have changed
            labels_new = self.predict_batchwise(model=model, dataloader=evalloader, device=device, use_P=True)
            if _are_labels_equal(labels_new=labels_new, labels_old=labels_old, threshold=tolerance_threshold):
                # training has converged
                # if debug:
                #    print("Clustering has converged")
                # break
                continue
            else:
                labels_old = labels_new.copy()

        # Extract P and m
        self.P = self.get_P()
        self.m = [len(P_i) for P_i in self.P]
        return model, self
        """


        # Deactivate Batchnorm and dropout
        model.eval()
        model.to(device)
        self.to_device(device)

        if trainloader is None and data is not None:
            trainloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        elif trainloader is None and data is None:
            raise ValueError("trainloader and data cannot be both None.")
        if evalloader is None and data is not None:
            # Evalloader is used for checking label change. Only difference to the trainloader here is that shuffle=False.
            evalloader = get_dataloader(data, batch_size=batch_size, shuffle=False, drop_last=False)

        if fix_rec_error:
            if debug: print("Calculate initial reconstruction error")
            _, _, init_rec_loss = enrc_encode_decode_batchwise_with_loss(V=self.V, centers=self.centers, model=model,
                                                                         dataloader=evalloader, device=device, loss_fn=loss_fn)
            # For numerical stability we add a small number
            init_rec_loss += 1e-8
            if debug: print("Initial reconstruction error is ", init_rec_loss)
        i = 0
        labels_old = None
        for epoch_i in range(max_epochs):
            for batch in trainloader:
                if self.augmentation_invariance:
                    batch_data_aug = batch[1].to(device)
                    batch_data = batch[2].to(device)
                else:
                    batch_data = batch[1].to(device)

                z = model.encode(batch_data)
                subspace_loss, z_rot, z_rot_back, assignment_matrix_dict = self(z)
                reconstruction = model.decode(z_rot_back)
                rec_loss = loss_fn(reconstruction, batch_data)

                if self.augmentation_invariance:
                    z_aug = model.encode(batch_data_aug)
                    # reuse assignments
                    subspace_loss_aug, _, z_rot_back_aug, _ = self(z_aug, assignment_matrix_dict=assignment_matrix_dict)
                    reconstruction_aug = model.decode(z_rot_back_aug)
                    rec_loss_aug = loss_fn(reconstruction_aug, batch_data_aug)
                    rec_loss = (rec_loss + rec_loss_aug) / 2
                    subspace_loss = (subspace_loss + subspace_loss_aug) / 2

                if fix_rec_error:
                    rec_weight = rec_loss.item() / init_rec_loss + subspace_loss.item() / rec_loss.item()
                    if rec_weight < 1:
                        rec_weight = 1.0
                    rec_loss *= rec_weight

                summed_loss = self.degree_of_space_distortion * subspace_loss + self.degree_of_space_preservation * rec_loss
                optimizer.zero_grad()
                summed_loss.backward()
                optimizer.step()

                # Update Assignments and Centroids on GPU
                with torch.no_grad():
                    self.update_centers(z_rot, assignment_matrix_dict)
                # Check if clusters have to be reinitialized
                for subspace_i in range(len(self.centers)):
                    reinit_centers(enrc=self, subspace_id=subspace_i, dataloader=trainloader, model=model,
                                   n_samples=512, kmeans_steps=10, debug=debug)

                # Increase reinit_threshold over time
                self.reinit_threshold = int(np.sqrt(i + 1))

                i += 1
            if (epoch_i - 1) % print_step == 0 or epoch_i == (max_epochs - 1):
                with torch.no_grad():
                    # Rotation loss is calculated to check if its deviation from an orthogonal matrix
                    rotation_loss = self.rotation_loss()
                    if debug:
                        print(
                            f"Epoch {epoch_i}/{max_epochs - 1}: summed_loss: {summed_loss.item():.4f}, "
                            f"subspace_losses: {subspace_loss.item():.4f}, rec_loss: {rec_loss.item():.4f},"
                            f"rotation_loss: rotation loss not calculated")

            if scheduler is not None:
                scheduler.step()

            if tolerance_threshold is not None and tolerance_threshold > 0:
                # Check if labels have changed
                labels_new = self.predict_batchwise(model=model, dataloader=evalloader, device=device, use_P=True)
                if _are_labels_equal(labels_new=labels_new, labels_old=labels_old, threshold=tolerance_threshold):
                    # training has converged
                    if debug:
                        print("Clustering has converged")
                    break
                else:
                    labels_old = labels_new.copy()

        # Extract P and m
        self.P = self.get_P()
        self.m = [len(P_i) for P_i in self.P]
        return model, self
"""
===================== Helper Functions =====================
"""


def beta_weights_init(P, n_dims, high_value=0.9):
    """Initializes parameters of the such that betas will be set randomly. See 2.2. Initialization and Augmentation
    Procedure in the paper for more details.

    Parameters
    ----------
    P : list, list containing projections for each subspace
    n_dims : int, dimensionality of the embedded data
    high_value : float, default=0.9, value that should be initially used to indicate strength of assignment of a specific dimension to the clustering.

    Returns
    ----------
    beta_weights : torch.tensor, intialized weights that are input in the softmax to get the betas.
    """
    print("init beta weights")
    print("P", P)
    print("n_dims", n_dims)
    print("high_value", high_value)

    n_sub_clusterings = len(P)
    assert 1.0 > high_value > 0.0, "high_value should be between 0 and 1"
    if n_sub_clusterings == 1:
        print("Single Subspace Detected, create all zero reconstruction space")
        beta_hard = np.zeros((n_sub_clusterings + 1, n_dims), dtype=np.float32)
    else:
        beta_hard = np.zeros((n_sub_clusterings, n_dims), dtype=np.float32)

    for sub_i, p in enumerate(P):
        for dim in p:
            beta_hard[sub_i, dim] = 1.0
    beta_hard_np = np.array(beta_hard, dtype=np.int)
    beta_soft_weights = np.array(beta_hard, dtype=np.float32)

    mask = beta_hard_np == 1
    beta_soft_weights[mask] = np.log(1/high_value - 1)

    mask = beta_hard_np == 0
    beta_soft_weights[mask] = np.log(1/(1.0-high_value) - 1)

    return torch.tensor(beta_soft_weights, dtype=torch.float32)

"""
===================== ACEDEC  =====================
"""

def _acedec(X, y, n_clusters, V, P, input_centers, batch_size, pretrain_learning_rate,
          learning_rate, pretrain_epochs, max_epochs, optimizer_class, clustering_optimizer_params, loss_fn,
          degree_of_space_distortion, degree_of_space_preservation, autoencoder, custom_trainloader, custom_testloader,
          embedding_size, init, random_state, device, scheduler, scheduler_params, tolerance_threshold, init_kwargs,
          init_subsample_size, debug, print_step, final_reclustering, cluster_assignment, cluster_assignment_kwargs,
          reclustering, reclustering_kwargs, update_cluster_centers, update_cluster_centers_kwargs,
          loss_calculation, loss_calculation_kwargs, ):
    print("setup acedec")
    # Set device to train on
    if device is None:
        device = detect_device()
    if random_state is None:
        random_state = np.random.RandomState
    # Setup dataloaders
    print("setup dataloaders")
    if custom_trainloader is None:
        trainloader = get_dataloader(X, batch_size=batch_size, additional_inputs=y, shuffle=True, drop_last=True)
    else:
        trainloader = custom_trainloader
    if custom_testloader is None:
        testloader = get_dataloader(X, batch_size=batch_size, additional_inputs=y, shuffle=False, drop_last=False)
    else:
        testloader = custom_testloader

    if trainloader.batch_size != batch_size:
        if debug: print("WARNING: Specified batch_size differs from trainloader.batch_size. Will use trainloader.batch_size.")
        batch_size = trainloader.batch_size
    # Use subsample of the data if specified
    print("subsample")
    print("size", init_subsample_size)
    """
    if init_subsample_size is not None and init_subsample_size > 0:
        rng = np.random.default_rng(random_state)
        rand_idx = rng.choice(X.shape[0], init_subsample_size, replace=False)
        subsampleloader = get_dataloader(X[rand_idx], batch_size=batch_size, additional_inputs=y[rand_idx], shuffle=False, drop_last=False)
        y = y[rand_idx]
    else:
        subsampleloader = testloader
    """
    if init_subsample_size is not None and init_subsample_size > 0 and init_subsample_size < X.shape[0]:
        rand_idx = random_state.choice(X.shape[0], init_subsample_size, replace=False)
        subsampleloader = get_dataloader(X[rand_idx], batch_size=batch_size, shuffle=False, drop_last=False)
        y = y[rand_idx]
    else:
        subsampleloader = testloader
    # Setup autoencoder
    print("setup autoencoder")
    autoencoder = get_trained_autoencoder(trainloader, pretrain_learning_rate, pretrain_epochs, device,
                                          optimizer_class, loss_fn, embedding_size, autoencoder)
    print("get embedded data")
    embedded_data = encode_batchwise(subsampleloader, autoencoder, device)

    input_centers, P, V, beta_weights = acedec_init(y=y, embedded_data=embedded_data,
                                                    n_clusters=n_clusters,
                                                    device=device, init=init,
                                                    rounds=10, epochs=10, batch_size=batch_size, debug=debug,
                                                    input_centers=input_centers, P=P, V=V, random_state=random_state,
                                                    max_iter=100, learning_rate=learning_rate,
                                                    optimizer_class=optimizer_class, init_kwargs=init_kwargs)

    # Run ACeDeC init
    print("Run ACeDeC init: ", init)
    if n_clusters[-1] != 1:
        raise ValueError("The last element of n_clusters must be 1. The noise space must be the last cluster.")
    if len(n_clusters) > 2:
        raise NotImplementedError("ACeDeC is currently only implemented for 2 subspaces. One cluster space and one "
                                  "Noise space")


    # Setup ACeDeC Module
    acedec_module = _ACeDeC_Module(input_centers, P, V, degree_of_space_distortion=degree_of_space_distortion,
                                   degree_of_space_preservation=degree_of_space_preservation,
                                   beta_weights=beta_weights, cluster_assignment=cluster_assignment,
                                   cluster_assignment_kwargs=cluster_assignment_kwargs,
                                   reclustering=reclustering, reclustering_kwargs=reclustering_kwargs,
                                   update_cluster_centers=update_cluster_centers,
                                   update_cluster_centers_kwargs=update_cluster_centers_kwargs,
                                   loss_calculation=loss_calculation,
                                   loss_calculation_kwargs=loss_calculation_kwargs).to_device(device)
    """
    param_dict = [{'params': autoencoder.parameters(),
                   'lr': learning_rate},
                  {'params': [acedec_module.V],
                   'lr': learning_rate},
                  # In accordance to the original paper we update the betas 10 times faster
                  {'params': [acedec_module.beta_weights],
                   'lr': learning_rate * 10},
                  ]
    """
    clustering_optimizer_beta_params = clustering_optimizer_params.copy()
    clustering_optimizer_beta_params["lr"] = clustering_optimizer_beta_params["lr"] * 10
    param_dict = [dict({'params': autoencoder.parameters()}, **clustering_optimizer_params),
                  dict({'params': [acedec_module.V]}, **clustering_optimizer_params),
                  dict({'params': [acedec_module.beta_weights]}, **clustering_optimizer_beta_params)
                  ]
    optimizer = optimizer_class(param_dict)

    if scheduler is not None:
        scheduler = scheduler(optimizer, **scheduler_params)

        # Training loop
    print("Start ACEDEC training")
    acedec_module.fit(trainloader=trainloader, evalloader=testloader,
                      data=X, labels=y,
                    max_epochs=max_epochs,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    batch_size=batch_size,
                    model=autoencoder,
                    device=device,
                    scheduler=scheduler,
                    tolerance_threshold=tolerance_threshold,
                    print_step=print_step,
                    debug=debug)


    cluster_labels_before_reclustering = acedec_module.predict_batchwise(model=autoencoder, dataloader=testloader,
                                                                       device=device, use_P=True)
    """
    if final_reclustering:
        acedec_module.recluster(y=y, dataloader=subsampleloader, model=autoencoder, device=device)
    # TODO: skip recluster call and do it outside in the example file
    # Predict labels and transfer other parameters to numpy
    cluster_labels = acedec_module.predict_batchwise(model=autoencoder, dataloader=testloader, device=device, use_P=True)[:, 0]
    print("CLUSTER LABELS ", cluster_labels)
    """
    # Recluster
    if final_reclustering:
        if debug:
            print("Recluster")
        acedec_module.recluster(y, dataloader=subsampleloader, model=autoencoder, device=device,
                                optimizer_params=clustering_optimizer_params,
                                optimizer_class=optimizer_class)
        # Predict labels and transfer other parameters to numpy
        cluster_labels = acedec_module.predict_batchwise(model=autoencoder, dataloader=testloader, device=device,
                                                         use_P=True)
        if debug:
            print("Betas after reclustering")
            print(acedec_module.subspace_betas().detach().cpu().numpy())
    else:
        cluster_labels = cluster_labels_before_reclustering
    cluster_centers = [centers_i.detach().cpu().numpy() for centers_i in acedec_module.centers]
    V = acedec_module.V.detach().cpu().numpy()
    betas = acedec_module.subspace_betas().detach().cpu().numpy()
    P = acedec_module.P
    m = acedec_module.m
    return cluster_labels, cluster_centers, V, m, betas, P, n_clusters, autoencoder, cluster_labels_before_reclustering


class ACEDEC(BaseEstimator, ClusterMixin):
    """Create an instance of the ACEDEC algorithm.

    Parameters
    ----------
    n_clusters : list, list containing number of clusters for each clustering
    V : numpy.ndarray, default=None, orthogonal rotation matrix (optional)
    P : list, default=None, list containing projections for each clustering (optional)
    input_centers : list, default=None, list containing the cluster centers for each clustering (optional)
    batch_size : int, default=128, size of the data batches
    pretrain_learning_rate : float, default=1e-3, learning rate for the pretraining of the autoencoder
    clustering_learning_rate : float, default=1e-4, learning rate of the actual clustering procedure
    pretrain_epochs : int, default=100, number of epochs for the pretraining of the autoencoder
    clustering_epochs : int, default=150, maximum number of epochs for the actual clustering procedure
    tolerance_threshold : float, default=None, tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
                        for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
                        will train as long as the labels are not changing anymore.
    optimizer_class : torch.optim, default=torch.optim.Adam, optimizer for pretraining and training.
    loss_fn : torch.nn, default=torch.nn.MSELoss(), loss function for the reconstruction.
    degree_of_space_distortion : float, default=1.0, weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure
    degree_of_space_preservation : float, default=1.0, weight of regularization loss term, e.g., reconstruction loss.
    autoencoder : the input autoencoder. If None a new autoencoder will be created and trained.
    embedding_size : int, default=20, size of the embedding within the autoencoder. Only used if autoencoder is None.
    init : str, default='nrkmeans', choose which initialization strategy should be used. Has to be one of 'nrkmeans', 'random' or 'sgd'.
    random_state : use a fixed random state to get a repeatable solution (optional)
    device : torch.device, default=None, if device is None then it will be checked whether a gpu is available or not.
    scheduler : torch.optim.lr_scheduler, default=None, learning rate scheduler that should be used.
    scheduler_params : dict, default=None, dictionary of the parameters of the scheduler object
    init_kwargs : dict, default=None, additional parameters that are used if init is a callable. (optional)
    init_subsample_size: int, default=None, specify if only a subsample of size 'init_subsample_size' of the data should be used for the initialization. (optional)
    debug: bool, default=False, if True additional information during the training will be printed.
    recluster: whether a final reclustering should be performed

    Raises
    ----------
    ValueError : if init is not one of 'nrkmeans', 'random', 'auto' or 'sgd'.

    References
    ----------
    Miklautz, Lukas and Bauer, Lena G. M. and Mautz, Dominik and Tschiatschek, Sebastian and Böhm, Christian and
    Plant, Claudia.  "Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces."
    Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21), 2826--2832, 19.-27.08.2021.
    """

    def __init__(self, n_clusters: Union[int, list], V=None, P=None, input_centers=None, batch_size=128,
                 pretrain_optimizer_params: dict = {"lr":1e-3, "epochs":100},
                 clustering_optimizer_params: dict = {"lr":1e-4, "epochs": 150},
                 tolerance_threshold=None,
                 optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss(),
                 degree_of_space_distortion=1.0, degree_of_space_preservation=1.0, autoencoder=None,
                 custom_dataloaders=[None, None],
                 embedding_size=20, init="acedec", random_state=None, device=None, scheduler=None,
                 scheduler_params=None, init_kwargs=None, init_subsample_size=None, debug=False, print_step=10,
                 cluster_assignment="acedec", cluster_assignment_kwargs=None,
                 reclustering="acedec", reclustering_kwargs=None, update_cluster_centers="acedec",
                 update_cluster_centers_kwargs=None, loss_calculation="acedec", loss_calculation_kwargs=None,
                 final_reclustering: bool = False):
        if type(n_clusters) == int:
            n_clusters = [n_clusters, 1]
        self.n_clusters = n_clusters.copy()
        self.random_state = check_random_state(random_state)
        set_torch_seed(self.random_state)
        self.device = device
        if self.device is None:
            self.device = detect_device()
        self.batch_size = batch_size
        self.pretrain_learning_rate = pretrain_optimizer_params["lr"]
        self.clustering_optimizer_params = clustering_optimizer_params
        self.clustering_learning_rate = clustering_optimizer_params["lr"]
        self.pretrain_epochs = pretrain_optimizer_params["epochs"]
        self.clustering_epochs = clustering_optimizer_params["epochs"]
        self.tolerance_threshold = tolerance_threshold
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.degree_of_space_distortion = degree_of_space_distortion
        self.degree_of_space_preservation = degree_of_space_preservation
        self.autoencoder = autoencoder
        self.custom_trainloader = custom_dataloaders[0]
        self.custom_testloader = custom_dataloaders[1]
        self.embedding_size = embedding_size
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.init_kwargs = init_kwargs
        self.init_subsample_size = init_subsample_size
        self.debug = debug
        self.print_step = print_step
        self.cluster_assignment = cluster_assignment
        self.cluster_assignment_kwargs = cluster_assignment_kwargs
        self.reclustering = reclustering
        self.reclustering_kwargs = reclustering_kwargs
        self.update_cluster_centers = update_cluster_centers
        self.update_cluster_centers_kwargs = update_cluster_centers_kwargs
        self.loss_calculation = loss_calculation
        self.loss_calculation_kwargs = loss_calculation_kwargs
        self.final_reclustering = final_reclustering

        print(n_clusters)
        if len(self.n_clusters) > 2:
            raise NotImplementedError(f"currently only two cluster spaces are supported, but {len(self.n_clusters)} "
                                      f"were "
                                      f"given. \n The noise space does need to be given as Input of ACeDeC")
        elif len(self.n_clusters) < 2:
            raise ValueError(f"At least one clustering and one noise space has to be given, but {len(self.n_clusters)} "
                             f"were given")
        #self.n_clusters.append(1)

        if init in available_init_strategies():
            self.init = init
        else:
            raise ValueError(f"init={init} does not exist, has to be one of {available_init_strategies()}.")
        self.input_centers = input_centers
        self.V = V
        self.m = None
        self.P = P

    def fit(self, X, y = None):
        """Cluster the input dataset with the ENRC algorithm. Saves the labels, centers, V, m, Betas, and P
        in the ENRC object.

        Parameters
        ----------
        X : np.ndarray, input data
        y : np.ndarray, default=None, ground truth labels. If given the labels will be used to fit the model. Missing labels can be given as -1.


        Returns
        ----------
        ENRC : returns the ENRC object
        """
        if y is None:
            y = np.zeros(X.shape[0])-1

        cluster_labels, cluster_centers, V, m, betas, P, n_clusters, autoencoder, cluster_labels_before_reclustering= _acedec(X=X, y=y,
                                                                                         n_clusters=self.n_clusters,
                                                                                         V=self.V,
                                                                                         P=self.P,
                                                                                         input_centers=self.input_centers,
                                                                                         batch_size=self.batch_size,
                                                                                         pretrain_learning_rate=self.pretrain_learning_rate,
                                                                                         learning_rate=self.clustering_learning_rate,
                                                                                         pretrain_epochs=self.pretrain_epochs,
                                                                                         max_epochs=self.clustering_epochs,
                                                                                         tolerance_threshold=self.tolerance_threshold,
                                                                                         optimizer_class=self.optimizer_class,
                                                                                         clustering_optimizer_params=self.clustering_optimizer_params,
                                                                                         loss_fn=self.loss_fn,
                                                                                         degree_of_space_distortion=self.degree_of_space_distortion,
                                                                                         degree_of_space_preservation=self.degree_of_space_preservation,
                                                                                         autoencoder=self.autoencoder,
                                                                                         custom_testloader=self.custom_testloader,
                                                                                         custom_trainloader=self.custom_trainloader,
                                                                                         embedding_size=self.embedding_size,
                                                                                         init=self.init,
                                                                                         random_state=self.random_state,
                                                                                         device=self.device,
                                                                                         scheduler=self.scheduler,
                                                                                         scheduler_params=self.scheduler_params,
                                                                                         init_kwargs=self.init_kwargs,
                                                                                         init_subsample_size=self.init_subsample_size,
                                                                                         debug=self.debug,
                                                                                         print_step=self.print_step,
                                                                                         cluster_assignment=self.cluster_assignment,
                                                                                         cluster_assignment_kwargs=self.cluster_assignment_kwargs,
                                                                                         reclustering=self.reclustering,
                                                                                         reclustering_kwargs=self.reclustering_kwargs,
                                                                                         update_cluster_centers=self.update_cluster_centers,
                                                                                         update_cluster_centers_kwargs=self.update_cluster_centers_kwargs,
                                                                                         loss_calculation=self.loss_calculation,
                                                                                         loss_calculation_kwargs=self.loss_calculation_kwargs,
                                                                                         final_reclustering=self.final_reclustering,
        )
        # Update class variables
        self.labels_ = cluster_labels[:, 0]
        self.cluster_centers_ = cluster_centers
        self.acedec_labels_ = cluster_labels_before_reclustering[:, 0]
        self.V = V
        self.m = m
        self.P = P
        self.betas = betas
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        return self

    def predict(self, X, use_P=True):
        """Predicts the labels for each clustering of X in a mini-batch manner.

        Parameters
        ----------
        X : np.ndarray, input data
        use_P: bool, default=True, if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used

        Returns
        -------
        predicted_labels : np.ndarray, n x c matrix, where n is the number of data points in X and c is the number of clusterings.
        """

        dataloader = get_dataloader(X, batch_size=self.batch_size, shuffle=False, drop_last=False)


        self.autoencoder.to(self.device)
        return enrc_predict_batchwise(V=torch.from_numpy(self.V).float().to(self.device),
                                      centers=[torch.from_numpy(c).float().to(self.device) for c in
                                               self.cluster_centers_],
                                      subspace_betas=torch.from_numpy(self.betas).float().to(self.device),
                                      model=self.autoencoder,
                                      dataloader=dataloader,
                                      device=self.device,
                                      use_P=use_P,
                                      )[:, 0]

    def transform_full_space(self, X, embedded=False):
        """
        Embedds the input dataset with the autoencoder and the matrix V from the ENRC object.
        Parameters
        ----------
        X : np.ndarray, input data
        embedded: bool, default=False, if True, then X is assumed to be already embedded.

        Returns
        -------
        The transformed data
        """
        if not embedded:
            dataloader = get_dataloader(X, batch_size=self.batch_size, shuffle=False, drop_last=False)
            emb = encode_batchwise(dataloader=dataloader, model=self.autoencoder, device=self.device)
        else:
            emb = X
        return np.matmul(emb, self.V)

    def transform_subspace(self, X, subspace_index, embedded=False):
        """
        Embedds the input dataset with the autoencoder and with the matrix V projected onto a special clusterspace_nr.

        Parameters
        ----------
        X : np.ndarray, input data
        subspace_index: int, index of the subspace_nr
        embedded: bool, default=False, if True, then X is assumed to be already embedded.

        Returns
        -------
        The transformed subspace
        """
        if not embedded:
            dataloader = get_dataloader(X, batch_size=self.batch_size, shuffle=False, drop_last=False)
            emb = encode_batchwise(dataloader=dataloader, model=self.autoencoder, device=self.device)
        else:
            emb = X
        cluster_space_V = self.V[:, self.P[subspace_index]]
        return np.matmul(emb, cluster_space_V)

    def plot_subspace(self, X, subspace_index, labels=None, plot_centers=False, title=None, gt=None, equal_axis=False):
        """
        Plot the specified subspace_nr as scatter matrix plot.

        Parameters
        ----------
        X : np.ndarray, input data
        subspace_index: int, index of the subspace_nr
        labels: np.array, default=None, the labels to use for the plot (default: labels found by Nr-Kmeans)
        plot_centers: boolean, default=False, plot centers if True
        title: string,  default=None, show title if defined
        gt: list, default=None, of ground truth labels
        equal_axis: boolean, default=False equalize axis if True
        Returns
        -------
        scatter matrix plot of the input data
        """
        if self.labels_ is None:
            raise Exception("The ACEDEC algorithm has not run yet. Use the fit() function first.")
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
