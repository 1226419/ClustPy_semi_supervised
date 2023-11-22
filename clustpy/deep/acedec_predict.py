import torch
from clustpy.deep.enrc import enrc_predict_batchwise, enrc_predict
from clustpy.deep.dec import _dec_predict
import numpy as np

def acedec_predict(z, V, centers, subspace_betas, use_P=False, prediction="acedec", prediction_kwargs=None):
    """Predicts the labels for each clustering of an input z. Ignores the noise space cluster.

    Parameters
    ----------
    z : torch.tensor, embedded input data point, can also be a mini-batch of embedded points
    V : torch.tensor, orthogonal rotation matrix
    centers : list of torch.tensors, cluster centers for each clustering
    subspace_betas : weights for each dimension per clustering. Calculated via softmax(beta_weights).
    use_P: bool, default=False, if True then P will be used to hard select the dimensions for each clustering, else the soft subspace_beta weights are used

    Returns
    -------
    predicted_labels : n x c matrix, where n is the number of data points in z and c is the number of clusterings.
    """
    if prediction == "acedec":
        return enrc_predict(z, V, centers[:-1], subspace_betas, use_P=use_P)
    elif prediction =="dec":
        if "alpha" in prediction_kwargs.keys():
            alpha = prediction_kwargs["alpha"]
        else:
            alpha = 0.5
            print("no alpha value found in prediction_kwargs - default to 0.5")
        if "feature_weights" in prediction_kwargs.keys():
            feature_weights = prediction_kwargs["feature_weights"]
        else:
            feature_weights = None
            print("no feature_weights value found in prediction_kwargs - default to None")
        return _dec_predict(centers[:-1], z, alpha, feature_weights)


def acedec_predict_batchwise(V, centers, subspace_betas, model, dataloader, device=torch.device("cpu"), use_P=False,
                             prediction="acedec", prediction_kwargs=None):
    """Predicts the labels for each clustering of a dataloader in a mini-batch manner.
        Ignores the noise space cluster

    Parameters
    ----------
    V : torch.tensor, orthogonal rotation matrix
    centers : list of torch.tensors, cluster centers for each clustering
    subspace_betas : weights for each dimension per clustering. Calculated via softmax(beta_weights).
    model : torch.nn.Module, the input model for encoding the data
    dataloader : torch.utils.data.DataLoader, dataloader to be used for prediction
    device : torch.device, default=torch.device('cpu'), device to be predicted on
    use_P: bool, default=False, if True then P will be used to hard select the dimensions for each clustering, else the soft beta weights are used

    Returns
    -------
    predicted_labels : n x c matrix, where n is the number of data points in z and c is the number of clusterings.
    """
    if prediction == "acedec":
        return enrc_predict_batchwise(V, centers[:-1], subspace_betas, model, dataloader, device=device, use_P=use_P)
    elif prediction =="dec":
        if "alpha" in prediction_kwargs.keys():
            alpha = prediction_kwargs["alpha"]
        else:
            alpha = 0.5
            print("no alpha value found in prediction_kwargs - default to 0.5")
        if "feature_weights" in prediction_kwargs.keys():
            feature_weights = prediction_kwargs["feature_weights"]
        else:
            feature_weights = None
            print("no feature_weights value found in prediction_kwargs - default to None")
        model.eval()
        predictions = []
        # TODO check if batchwise predict is correct
        with torch.no_grad():
            for batch in dataloader:
                batch_data = batch[1].to(device)
                z = model.encode(batch_data)
                pred_i = _dec_predict(centers[:-1], z, alpha, feature_weights)
                predictions.append(pred_i)
        return np.concatenate(predictions)
