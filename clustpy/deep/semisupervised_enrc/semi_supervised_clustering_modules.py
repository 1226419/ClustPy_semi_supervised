import torch
import numpy as np
from clustpy.deep._utils import int_to_one_hot, squared_euclidean_distance, encode_batchwise
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep.semisupervised_enrc.helper_functions import _get_P, _rotate, _rotate_back, enrc_predict, \
     enrc_encode_decode_batchwise_with_loss, enrc_predict_batchwise, _are_labels_equal
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_reclustering_reinit import reinit_centers
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_init_betas import beta_weights_init

import torch
import numpy as np
from clustpy.deep._utils import int_to_one_hot, squared_euclidean_distance, encode_batchwise
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep.semisupervised_enrc.helper_functions import _get_P, _rotate, _rotate_back, enrc_predict, \
     enrc_encode_decode_batchwise_with_loss, enrc_predict_batchwise, _are_labels_equal
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_reclustering_reinit import reinit_centers
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_init_betas import beta_weights_init
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_module import _ENRC_Module


class _Label_Loss_Module_based_on_ENRC(_ENRC_Module):
    """
    The ENRC torch.nn.Module.

    Parameters
    ----------
    centers : list
        list containing the cluster centers for each clustering
    P : list
        list containing projections for each clustering
    V : np.ndarray
        orthogonal rotation matrix
    beta_init_value : float
        initial values of beta weights. Is ignored if beta_weights is not None (default: 0.9)
    degree_of_space_distortion : float
        weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure (default: 1.0)
    degree_of_space_preservation : float
        weight of regularization loss term, e.g., reconstruction loss (default: 1.0)
    center_lr : float
        weight for updating the centers via mini-batch k-means. Has to be set between 0 and 1. If set to 1.0 than only the mini-batch centroid will be used,
        neglecting the past state and if set to 0 then no update is happening (default: 0.5)
    rotate_centers : bool
        if True then centers are multiplied with V before they are used, because ENRC assumes that the centers lie already in the V-rotated space (default: False)
    beta_weights : np.ndarray
        initial beta weights for the softmax (optional). If not None, then beta_init_value will be ignored (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    lonely_centers_count : list
        list of np.ndarrays, count indicating how often a center in a clustering has not received any updates, because no points were assigned to it.
        The lonely_centers_count of a center is reset if it has been reinitialized.
    mask_sum : list
        list of torch.tensors, contains the average number of points assigned to each cluster in each clustering over the training.
    reinit_threshold : int
        threshold that indicates when a cluster should be reinitialized. Starts with 1 and increases during training with int(np.sqrt(i+1)), where i is the number of mini-batch iterations.
    augmentation_invariance : bool (default: False)

    Raises
    ----------
    ValueError : if center_lr is not in [0,1]
    """

    def __init__(self, centers: list, P: list, V: np.ndarray, beta_init_value: float = 0.9,
                 degree_of_space_distortion: float = 1.0, degree_of_space_preservation: float = 1.0,
                 center_lr: float = 0.5, rotate_centers: bool = False, beta_weights: np.ndarray = None, augmentation_invariance: bool = False):
        super().__init__(centers, P, V, beta_init_value,
                 degree_of_space_distortion, degree_of_space_preservation,
                 center_lr, rotate_centers, beta_weights, augmentation_invariance)

    def forward(self, z: torch.Tensor, assignment_matrix_dict: dict = None, batch_label_data: torch.Tensor = None,
                epoch_i: int = None) -> (torch.Tensor, torch.Tensor, torch.Tensor, dict):
        """
        Calculates the k-means loss and cluster assignments for each clustering.

        Parameters
        ----------
        z : torch.Tensor
            embedded input data point, can also be a mini-batch of embedded points
        assignment_matrix_dict : dict
            dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments (default: None)
        batch_label_data : torch.Tensor
            labeled_data of the current data points, can also be a mini-batch of embedded points
        epoch_i : int
            current epoch that calls the forward function


        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor, dict)
            averaged sum of all k-means losses for each clustering,
            the rotated embedded point,
            the back rotated embedded point,
            dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments
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

        for i, centers_i in enumerate(self.centers):
            weighted_squared_diff = squared_euclidean_distance(z_rot, centers_i.detach(), weights=subspace_betas[i, :])
            weighted_squared_diff /= z_rot.shape[0]

            if overwrite_assignments:
                assignments = weighted_squared_diff.detach().argmin(1)
                one_hot_mask = int_to_one_hot(assignments, centers_i.shape[0])
                assignment_matrix_dict[i] = one_hot_mask
            else:
                one_hot_mask = assignment_matrix_dict[i]

            if (batch_label_data is not None) and (len(centers_i) > 1):
                # find which label belongs to which cluster center -> for now we just force the centers to get the right
                # ordering as well -> lets see what happens if we use unlabeled init
                """
                print("hola")
                detached_weighted_sq_diff = weighted_squared_diff.detach()
                current_labels_mask = (batch_label_data != -1).int()
                labeled_weigthed_sq_diff = detached_weighted_sq_diff[current_labels_mask == 1]
                only_labeld_labels = batch_label_data[current_labels_mask == 1]
                only_labeld_labels_one_hot = int_to_one_hot(only_labeld_labels, centers_i.shape[0])
                awhat = labeled_weigthed_sq_diff * only_labeld_labels_one_hot
                """
                # assign all labeld points to their correct cluster center even if it is not the closest.
                # alternative version -> only penalise them slightly
                # alternative version -> in the beginning assign them all to the correct one later be less strict, allow flexibility..
                current_labels_mask = (batch_label_data != -1).int()
                only_labeld_labels = batch_label_data[current_labels_mask == 1]
                only_labeld_labels_one_hot = int_to_one_hot(only_labeld_labels, centers_i.shape[0])
                one_hot_mask[current_labels_mask == 1] = only_labeld_labels_one_hot
            weighted_squared_diff_masked = weighted_squared_diff * one_hot_mask
            subspace_losses += weighted_squared_diff_masked.sum()

        subspace_losses = subspace_losses / subspace_betas.shape[0]
        return subspace_losses, z_rot, z_rot_back, assignment_matrix_dict

    def fit(self, trainloader: torch.utils.data.DataLoader, evalloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, max_epochs: int, model: torch.nn.Module,
            batch_size: int, loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
            device: torch.device = torch.device("cpu"), print_step: int = 5, debug: bool = True,
            scheduler: torch.optim.lr_scheduler = None, fix_rec_error: bool = False,
            tolerance_threshold: float = None, data: torch.Tensor = None, y: torch.Tensor = None) -> (torch.nn.Module, '_ENRC_Module'):
        """
        Trains ENRC and the autoencoder in place.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        evalloader : torch.utils.data.DataLoader
            Evalloader is used for checking label change
        optimizer : torch.optim.Optimizer
            parameterized optimizer to be used
        max_epochs : int
            maximum number of epochs for training
        model : torch.nn.Module
            The underlying autoencoder
        batch_size: int
            batch size for dataloader
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction (default: torch.nn.MSELoss())
        device : torch.device
            device to be trained on (default: torch.device('cpu'))
        print_step : int
            specifies how often the losses are printed (default: 5)
        debug : bool
            if True than training errors will be printed (default: True)
        scheduler : torch.optim.lr_scheduler
            parameterized learning rate scheduler that should be used (default: None)
        fix_rec_error : bool
            if set to True than reconstruction loss is weighted proportionally to the cluster loss. Only used for init='sgd' (default: False)
        tolerance_threshold : float
            tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
            for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
            will train as long as max_epochs (default: None)
        data : torch.Tensor / np.ndarray
            dataset to be used for training (default: None)
        Returns
        -------
        tuple : (torch.nn.Module, _ENRC_Module)
            trained autoencoder,
            trained enrc module
        """

        # Deactivate Batchnorm and dropout
        model.eval()
        model.to(device)
        self.to_device(device)
        print("y", y)

        if trainloader is None and data is not None:
            trainloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        elif trainloader is None and data is None:
            raise ValueError("trainloader and data cannot be both None.")
        if evalloader is None and data is not None:
            # Evalloader is used for checking label change. Only difference to the trainloader here is that shuffle=False.
            evalloader = get_dataloader(data, batch_size=batch_size, shuffle=False, drop_last=False)

        if fix_rec_error:
            if debug: print("Calculate initial reconstruction error")
            _, _, init_rec_loss = enrc_encode_decode_batchwise_with_loss(V=self.V, centers=self.centers, model=model, dataloader=evalloader, device=device, loss_fn=loss_fn)
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
                batch_label_data = batch[-1].to(device)

                assert batch_label_data.shape != batch_data.shape, "no label data was passed from dataloader"

                z = model.encode(batch_data)
                subspace_loss, z_rot, z_rot_back, assignment_matrix_dict = self(z, batch_label_data=batch_label_data,
                                                                                epoch_i=epoch_i)
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
                    rec_weight = rec_loss.item()/init_rec_loss + subspace_loss.item()/rec_loss.item()
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
                        print(f"Epoch {epoch_i}/{max_epochs - 1}: summed_loss: {summed_loss.item():.4f}, subspace_losses: {subspace_loss.item():.4f}, rec_loss: {rec_loss.item():.4f}, rotation_loss: {rotation_loss.item():.4f}")

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


class _Label_Loss_Module_based_on_ENRC_delayed(_ENRC_Module):
    """
    The ENRC torch.nn.Module.

    Parameters
    ----------
    centers : list
        list containing the cluster centers for each clustering
    P : list
        list containing projections for each clustering
    V : np.ndarray
        orthogonal rotation matrix
    beta_init_value : float
        initial values of beta weights. Is ignored if beta_weights is not None (default: 0.9)
    degree_of_space_distortion : float
        weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure (default: 1.0)
    degree_of_space_preservation : float
        weight of regularization loss term, e.g., reconstruction loss (default: 1.0)
    center_lr : float
        weight for updating the centers via mini-batch k-means. Has to be set between 0 and 1. If set to 1.0 than only the mini-batch centroid will be used,
        neglecting the past state and if set to 0 then no update is happening (default: 0.5)
    rotate_centers : bool
        if True then centers are multiplied with V before they are used, because ENRC assumes that the centers lie already in the V-rotated space (default: False)
    beta_weights : np.ndarray
        initial beta weights for the softmax (optional). If not None, then beta_init_value will be ignored (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    lonely_centers_count : list
        list of np.ndarrays, count indicating how often a center in a clustering has not received any updates, because no points were assigned to it.
        The lonely_centers_count of a center is reset if it has been reinitialized.
    mask_sum : list
        list of torch.tensors, contains the average number of points assigned to each cluster in each clustering over the training.
    reinit_threshold : int
        threshold that indicates when a cluster should be reinitialized. Starts with 1 and increases during training with int(np.sqrt(i+1)), where i is the number of mini-batch iterations.
    augmentation_invariance : bool (default: False)

    Raises
    ----------
    ValueError : if center_lr is not in [0,1]
    """

    def __init__(self, centers: list, P: list, V: np.ndarray, beta_init_value: float = 0.9,
                 degree_of_space_distortion: float = 1.0, degree_of_space_preservation: float = 1.0,
                 center_lr: float = 0.5, rotate_centers: bool = False, beta_weights: np.ndarray = None, augmentation_invariance: bool = False):
        super().__init__(centers, P, V, beta_init_value,
                 degree_of_space_distortion, degree_of_space_preservation,
                 center_lr, rotate_centers, beta_weights, augmentation_invariance)

    def forward(self, z: torch.Tensor, assignment_matrix_dict: dict = None, batch_label_data: torch.Tensor = None,
                epoch_i: int = None) -> (torch.Tensor, torch.Tensor, torch.Tensor, dict):
        """
        Calculates the k-means loss and cluster assignments for each clustering.

        Parameters
        ----------
        z : torch.Tensor
            embedded input data point, can also be a mini-batch of embedded points
        assignment_matrix_dict : dict
            dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments (default: None)
        batch_label_data : torch.Tensor
            labeled_data of the current data points, can also be a mini-batch of embedded points
        epoch_i : int
            current epoch that calls the forward function


        Returns
        -------
        tuple : (torch.Tensor, torch.Tensor, torch.Tensor, dict)
            averaged sum of all k-means losses for each clustering,
            the rotated embedded point,
            the back rotated embedded point,
            dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments
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

        for i, centers_i in enumerate(self.centers):
            weighted_squared_diff = squared_euclidean_distance(z_rot, centers_i.detach(), weights=subspace_betas[i, :])
            weighted_squared_diff /= z_rot.shape[0]

            if overwrite_assignments:
                assignments = weighted_squared_diff.detach().argmin(1)
                one_hot_mask = int_to_one_hot(assignments, centers_i.shape[0])
                assignment_matrix_dict[i] = one_hot_mask
            else:
                one_hot_mask = assignment_matrix_dict[i]

            if (batch_label_data is not None) and (len(centers_i) > 1) and (epoch_i < 20):
                # find which label belongs to which cluster center -> for now we just force the centers to get the right
                # ordering as well -> lets see what happens if we use unlabeled init
                """
                detached_weighted_sq_diff = weighted_squared_diff.detach()
                current_labels_mask = (batch_label_data != -1).int()
                labeled_weigthed_sq_diff = detached_weighted_sq_diff[current_labels_mask == 1]
                only_labeld_labels = batch_label_data[current_labels_mask == 1]
                only_labeld_labels_one_hot = int_to_one_hot(only_labeld_labels, centers_i.shape[0])
                awhat = labeled_weigthed_sq_diff * only_labeld_labels_one_hot
                """
                # assign all labeld points to their correct cluster center even if it is not the closest.
                # -> in the beginning assign them all to the correct one later be less strict, allow flexibility..
                current_labels_mask = (batch_label_data != -1).int()
                only_labeld_labels = batch_label_data[current_labels_mask == 1]
                only_labeld_labels_one_hot = int_to_one_hot(only_labeld_labels, centers_i.shape[0])
                one_hot_mask[current_labels_mask == 1] = only_labeld_labels_one_hot
            weighted_squared_diff_masked = weighted_squared_diff * one_hot_mask
            subspace_losses += weighted_squared_diff_masked.sum()

        subspace_losses = subspace_losses / subspace_betas.shape[0]
        return subspace_losses, z_rot, z_rot_back, assignment_matrix_dict

    def fit(self, trainloader: torch.utils.data.DataLoader, evalloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, max_epochs: int, model: torch.nn.Module,
            batch_size: int, loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
            device: torch.device = torch.device("cpu"), print_step: int = 5, debug: bool = True,
            scheduler: torch.optim.lr_scheduler = None, fix_rec_error: bool = False,
            tolerance_threshold: float = None, data: torch.Tensor = None, y: torch.Tensor = None) -> (torch.nn.Module, '_ENRC_Module'):
        """
        Trains ENRC and the autoencoder in place.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        evalloader : torch.utils.data.DataLoader
            Evalloader is used for checking label change
        optimizer : torch.optim.Optimizer
            parameterized optimizer to be used
        max_epochs : int
            maximum number of epochs for training
        model : torch.nn.Module
            The underlying autoencoder
        batch_size: int
            batch size for dataloader
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction (default: torch.nn.MSELoss())
        device : torch.device
            device to be trained on (default: torch.device('cpu'))
        print_step : int
            specifies how often the losses are printed (default: 5)
        debug : bool
            if True than training errors will be printed (default: True)
        scheduler : torch.optim.lr_scheduler
            parameterized learning rate scheduler that should be used (default: None)
        fix_rec_error : bool
            if set to True than reconstruction loss is weighted proportionally to the cluster loss. Only used for init='sgd' (default: False)
        tolerance_threshold : float
            tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
            for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
            will train as long as max_epochs (default: None)
        data : torch.Tensor / np.ndarray
            dataset to be used for training (default: None)
        Returns
        -------
        tuple : (torch.nn.Module, _ENRC_Module)
            trained autoencoder,
            trained enrc module
        """

        # Deactivate Batchnorm and dropout
        model.eval()
        model.to(device)
        self.to_device(device)
        print("y", y)

        if trainloader is None and data is not None:
            trainloader = get_dataloader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        elif trainloader is None and data is None:
            raise ValueError("trainloader and data cannot be both None.")
        if evalloader is None and data is not None:
            # Evalloader is used for checking label change. Only difference to the trainloader here is that shuffle=False.
            evalloader = get_dataloader(data, batch_size=batch_size, shuffle=False, drop_last=False)

        if fix_rec_error:
            if debug: print("Calculate initial reconstruction error")
            _, _, init_rec_loss = enrc_encode_decode_batchwise_with_loss(V=self.V, centers=self.centers, model=model, dataloader=evalloader, device=device, loss_fn=loss_fn)
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
                batch_label_data = batch[-1].to(device)

                assert batch_label_data.shape != batch_data.shape, "no label data was passed from dataloader"

                z = model.encode(batch_data)
                subspace_loss, z_rot, z_rot_back, assignment_matrix_dict = self(z, batch_label_data=batch_label_data,
                                                                                epoch_i=epoch_i)
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
                    rec_weight = rec_loss.item()/init_rec_loss + subspace_loss.item()/rec_loss.item()
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
                        print(f"Epoch {epoch_i}/{max_epochs - 1}: summed_loss: {summed_loss.item():.4f}, subspace_losses: {subspace_loss.item():.4f}, rec_loss: {rec_loss.item():.4f}, rotation_loss: {rotation_loss.item():.4f}")

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


class _Label_Cluster_Assignment_Module(_ENRC_Module):
    """
    The ENRC torch.nn.Module.

    Parameters
    ----------
    centers : list
        list containing the cluster centers for each clustering
    P : list
        list containing projections for each clustering
    V : np.ndarray
        orthogonal rotation matrix
    beta_init_value : float
        initial values of beta weights. Is ignored if beta_weights is not None (default: 0.9)
    degree_of_space_distortion : float
        weight of the cluster loss term. The higher it is set the more the embedded space will be shaped to the assumed cluster structure (default: 1.0)
    degree_of_space_preservation : float
        weight of regularization loss term, e.g., reconstruction loss (default: 1.0)
    center_lr : float
        weight for updating the centers via mini-batch k-means. Has to be set between 0 and 1. If set to 1.0 than only the mini-batch centroid will be used,
        neglecting the past state and if set to 0 then no update is happening (default: 0.5)
    rotate_centers : bool
        if True then centers are multiplied with V before they are used, because ENRC assumes that the centers lie already in the V-rotated space (default: False)
    beta_weights : np.ndarray
        initial beta weights for the softmax (optional). If not None, then beta_init_value will be ignored (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    lonely_centers_count : list
        list of np.ndarrays, count indicating how often a center in a clustering has not received any updates, because no points were assigned to it.
        The lonely_centers_count of a center is reset if it has been reinitialized.
    mask_sum : list
        list of torch.tensors, contains the average number of points assigned to each cluster in each clustering over the training.
    reinit_threshold : int
        threshold that indicates when a cluster should be reinitialized. Starts with 1 and increases during training with int(np.sqrt(i+1)), where i is the number of mini-batch iterations.
    augmentation_invariance : bool (default: False)

    Raises
    ----------
    ValueError : if center_lr is not in [0,1]
    """

    def __init__(self, centers: list, P: list, V: np.ndarray, beta_init_value: float = 0.9,
                 degree_of_space_distortion: float = 1.0, degree_of_space_preservation: float = 1.0,
                 center_lr: float = 0.5, rotate_centers: bool = False, beta_weights: np.ndarray = None, augmentation_invariance: bool = False):
        super().__init__(centers, P, V, beta_init_value,
                 degree_of_space_distortion, degree_of_space_preservation,
                 center_lr, rotate_centers, beta_weights, augmentation_invariance)

    def update_center(self, data: torch.Tensor, one_hot_mask: torch.Tensor, subspace_id: int) -> None:
        """
        Inplace update of centers of a clusterings in subspace=subspace_id in a mini-batch fashion.

        Parameters
        ----------
        data : torch.Tensor
            data points, can also be a mini-batch of points
        one_hot_mask : torch.Tensor
            one hot encoded matrix of cluster assignments
        subspace_id : int
            integer which indicates which subspace the cluster to be updated are in

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

    def update_centers(self, z_rot: torch.Tensor, assignment_matrix_dict: dict) -> None:
        """
        Inplace update of all centers in all clusterings in a mini-batch fashion.

        Parameters
        ----------
        z_rot : torch.Tensor
            rotated data point, can also be a mini-batch of points
        assignment_matrix_dict : dict
            dict of torch.tensors, contains for each i^th clustering a one hot encoded matrix of cluster assignments
        """
        for subspace_i in range(len(self.centers)):
            self.update_center(z_rot.detach(),
                               assignment_matrix_dict[subspace_i],
                               subspace_id=subspace_i)
            

    def fit(self, trainloader: torch.utils.data.DataLoader, evalloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, max_epochs: int, model: torch.nn.Module,
            batch_size: int, loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
            device: torch.device = torch.device("cpu"), print_step: int = 5, debug: bool = True,
            scheduler: torch.optim.lr_scheduler = None, fix_rec_error: bool = False,
            tolerance_threshold: float = None, data: torch.Tensor = None, y: torch.Tensor = None) -> (torch.nn.Module, '_ENRC_Module'):
        """
        Trains ENRC and the autoencoder in place.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        evalloader : torch.utils.data.DataLoader
            Evalloader is used for checking label change
        optimizer : torch.optim.Optimizer
            parameterized optimizer to be used
        max_epochs : int
            maximum number of epochs for training
        model : torch.nn.Module
            The underlying autoencoder
        batch_size: int
            batch size for dataloader
        loss_fn : torch.nn.modules.loss._Loss
            loss function to be used for reconstruction (default: torch.nn.MSELoss())
        device : torch.device
            device to be trained on (default: torch.device('cpu'))
        print_step : int
            specifies how often the losses are printed (default: 5)
        debug : bool
            if True than training errors will be printed (default: True)
        scheduler : torch.optim.lr_scheduler
            parameterized learning rate scheduler that should be used (default: None)
        fix_rec_error : bool
            if set to True than reconstruction loss is weighted proportionally to the cluster loss. Only used for init='sgd' (default: False)
        tolerance_threshold : float
            tolerance threshold to determine when the training should stop. If the NMI(old_labels, new_labels) >= (1-tolerance_threshold)
            for all clusterings then the training will stop before max_epochs is reached. If set high than training will stop earlier then max_epochs, and if set to 0 or None the training
            will train as long as max_epochs (default: None)
        data : torch.Tensor / np.ndarray
            dataset to be used for training (default: None)
        Returns
        -------
        tuple : (torch.nn.Module, _ENRC_Module)
            trained autoencoder,
            trained enrc module
        """

        # Deactivate Batchnorm and dropout
        model.eval()
        model.to(device)
        self.to_device(device)
        print("y", y)

        if fix_rec_error:
            if debug: print("Calculate initial reconstruction error")
            _, _, init_rec_loss = enrc_encode_decode_batchwise_with_loss(V=self.V, centers=self.centers, model=model, dataloader=evalloader, device=device, loss_fn=loss_fn)
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
                batch_label_data = batch[-1].to(device)

                assert batch_label_data.shape != batch_data.shape, "no label data was passed from dataloader"

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
                    rec_weight = rec_loss.item()/init_rec_loss + subspace_loss.item()/rec_loss.item()
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
                        print(f"Epoch {epoch_i}/{max_epochs - 1}: summed_loss: {summed_loss.item():.4f}, subspace_losses: {subspace_loss.item():.4f}, rec_loss: {rec_loss.item():.4f}, rotation_loss: {rotation_loss.item():.4f}")

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
