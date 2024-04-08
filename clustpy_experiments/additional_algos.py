import numpy as np
import torch
from clustpy.deep import encode_batchwise, detect_device, get_dataloader
from clustpy.deep._utils import embedded_kmeans_prediction
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from clustpy.partition import SubKmeans


class AEKmeans():

    def __init__(self, n_clusters, autoencoder=None, batch_size=256, random_state: np.random.RandomState = None,
                 custom_dataloaders: tuple = None):
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if self.custom_dataloaders is None:
            testloader = get_dataloader(X,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        drop_last=False)
        else:
            _, testloader = self.custom_dataloaders
        X_ae = encode_batchwise(testloader, self.autoencoder, torch.device('cpu'))
        km = KMeans(self.n_clusters)
        km.fit(X_ae)
        self.labels_ = km.labels_
        self.cluster_centers_ = km.cluster_centers_

    def predict(self, X: np.ndarray) -> np.ndarray:
        dataloader = get_dataloader(X, self.batch_size, False, False)
        ae = self.autoencoder.to(detect_device())
        predicted_labels = embedded_kmeans_prediction(dataloader, self.cluster_centers_, ae)
        return predicted_labels


class AESubKmeans():

    def __init__(self, n_clusters, autoencoder=None, batch_size=256, random_state: np.random.RandomState = None,
                 custom_dataloaders: tuple = None):
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if self.custom_dataloaders is None:
            testloader = get_dataloader(X,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        drop_last=False)
        else:
            _, testloader = self.custom_dataloaders
        X_ae = encode_batchwise(testloader, self.autoencoder, torch.device('cpu'))
        sk = SubKmeans(self.n_clusters)
        sk.fit(X_ae)
        self.labels_ = sk.labels_


class AESpectral():

    def __init__(self, n_clusters, autoencoder=None, batch_size=256, random_state: np.random.RandomState = None,
                 custom_dataloaders: tuple = None):
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if self.custom_dataloaders is None:
            testloader = get_dataloader(X,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        drop_last=False)
        else:
            _, testloader = self.custom_dataloaders
        X_ae = encode_batchwise(testloader, self.autoencoder, torch.device('cpu'))
        sc = SpectralClustering(self.n_clusters)
        sc.fit(X_ae)
        self.labels_ = sc.labels_


class AEEM():

    def __init__(self, n_clusters, autoencoder=None, batch_size=256, random_state: np.random.RandomState = None,
                 custom_dataloaders: tuple = None):
        self.n_clusters = n_clusters
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.random_state = check_random_state(random_state)
        self.custom_dataloaders = custom_dataloaders

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        if self.custom_dataloaders is None:
            testloader = get_dataloader(X,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        drop_last=False)
        else:
            _, testloader = self.custom_dataloaders
        X_ae = encode_batchwise(testloader, self.autoencoder, torch.device('cpu'))
        gmm = GaussianMixture(self.n_clusters, n_init=3)
        gmm.fit(X_ae)
        self.labels_ = gmm.predict(X_ae)

