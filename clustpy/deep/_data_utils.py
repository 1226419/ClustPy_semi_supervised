import torch
import numpy as np


class _ClustpyDataset(torch.utils.data.Dataset):
    """
    Dataset wrapping tensors that has the indices always in the first entry.
    Each sample will be retrieved by indexing tensors along the first dimension.
    Implementation is based on torch.utils.data.TensorDataset.

    Parameters
    ----------
    *tensors : torch.Tensor
        tensors that have the same size of the first dimension. Usually contains the data.

    Attributes
    ----------
    tensors : torch.Tensor
        tensors that have the same size of the first dimension. Usually contains the data.
    """

    def __init__(self, *tensors: torch.Tensor):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index: int) -> tuple:
        """
        Get sample at specified index.

        Parameters
        ----------
        index : int
            index of the desired sample

        Returns
        -------
        final_tuple : tuple
            Tuple containing the sample. Consists of (index, data1, data2, ...), depending on the input tensors.
        """
        final_tuple = tuple([index] + [tensor[index] for tensor in self.tensors])
        return final_tuple

    def __len__(self) -> int:
        """
        Get length of the dataset which equals the length of the input tensors.

        Returns
        -------
        dataset_size : int
            Length of the dataset.
        """
        dataset_size = self.tensors[0].size(0)
        return dataset_size


def get_dataloader(X: np.ndarray, batch_size: int, shuffle: bool = True, drop_last: bool = False,
                   additional_inputs: list = None, dataset_class: torch.utils.data.Dataset = _ClustpyDataset,
                   **dl_kwargs: any) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for Deep Clustering algorithms.
    First entry always contains the indices of the data samples.
    Second entry always contains the actual data samples.
    If for example labels are desired, they can be passed through the additional_inputs parameter (should be a list).
    Other customizations (e.g. augmentation) can be implemented using a custom dataset_class.
    This custom class should stick to the conventions, [index, data, ...].

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the actual data set (can be np.ndarray or torch.Tensor)
    batch_size : int
        the batch size
    shuffle : bool
        boolean that defines if the data set should be shuffled (default: True)
    drop_last : bool
        boolean that defines if the last batch should be ignored (default: False)
    additional_inputs : list / np.ndarray / torch.Tensor
        additional inputs for the dataloader, e.g. labels. Can be None, np.ndarray, torch.Tensor or a list containing np.ndarrays/torch.Tensors (default: None)
    dataset_class : torch.utils.data.Dataset
        defines the class of the tensor dataset that is contained in the dataloader (default: _ClustpyDataset)
    dl_kwargs : any
        other arguments for torch.utils.data.DataLoader

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        The final dataloader
    """
    assert type(X) in [np.ndarray, torch.Tensor], "X must be of type np.ndarray or torch.Tensor."
    assert additional_inputs is None or type(additional_inputs) in [np.ndarray, torch.Tensor,
                                                                    list], "additional_input must be None or of type np.ndarray, torch.Tensor or list."
    check_if_data_is_normalized(X)
    if type(X) is np.ndarray:
        # Convert np.ndarray to torch.Tensor
        X = torch.from_numpy(X).float()
    dataset_input = [X]
    if additional_inputs is not None:
        # Check type of additional_inputs
        if type(additional_inputs) is np.ndarray:
            dataset_input.append(torch.from_numpy(additional_inputs).float())
        elif type(additional_inputs) is torch.Tensor:
            dataset_input.append(additional_inputs)
        else:
            for input in additional_inputs:
                if type(input) is np.ndarray:
                    input = torch.from_numpy(input).float()
                elif type(input) is not torch.Tensor:
                    raise Exception(
                        "inputs of additional_inputs must be of type np.ndarray or torch.Tensor. Your input type: {0}".format(
                            type(input)))
                dataset_input.append(input)
    dataset = dataset_class(*dataset_input)
    # Create dataloader using the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **dl_kwargs)
    return dataloader


def check_if_data_is_normalized(X):

    if X.min() < 0.0:
        print("negative data values found")

    if (X.max() <= 1.0) and (X.min() >= 0.0):
        print("probable MinMax normalized data detected")
        return True

    if (X.mean() <= 0.1) and (X.mean() >= -0.1) and (X.std() >= 0.9) and (X.std() <= 1.1):
        print("probable z-normalized data detected")
        return True

    if (X.max() > 100) and (X.min() == 0.0):
        print("Data was probably not normalised - make sure you performed a normalisation")
        return False

    print("Normalization check was inconclusive - make sure your data is normalised")
    print("mean", X.mean())
    print("std", X.std())
    print("max", X.max())
    print("min", X.min())
    return False
