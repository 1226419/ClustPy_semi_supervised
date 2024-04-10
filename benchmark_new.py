
from clustpy.data import load_optdigits, load_mnist, load_fmnist, load_usps, load_kmnist, load_cifar10, load_imagenet10, \
    load_imagenet_dog
from clustpy.deep import detect_device, get_dataloader, DEC, DCN, IDEC, ACeDeC
from clustpy.utils import EvaluationAlgorithm, EvaluationMetric
from clustpy.metrics import unsupervised_clustering_accuracy as acc, \
    information_theoretic_external_cluster_validity_measure as dom
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as ami, \
    adjusted_rand_score as ari

from clustpy_experiments.additional_algos import AEKmeans
from clustpy_experiments.experiment_learning_rate import experiment_learning_rate
from clustpy_experiments.experiment_initial_clustering import experiment_initial_clustering
from clustpy_experiments.experiment_feedforward_test_train import experiment_feedforward_test_train, experiment_feedforward_512_256_128_10
from clustpy_experiments.experiment_conv_resnet import experiment_convolutional_resnet18

from clustpy.deep.semisupervised_enrc.semi_supervised_acedec import ACeDeC as My_ACeDeC

DOWNLOAD_PATH = "Downloaded_datasets"
SAVE_DIR = "MyBenchmark/"


def _get_dataset_loaders():
    datasets = [
        #("Optdigits", load_optdigits),
        #("USPS", load_usps),
        ("MNIST", load_mnist),
        #("FMNIST", load_fmnist),
        #("KMNIST", load_kmnist),
        #("CIFAR10", load_cifar10),
        #("ImageNet10", load_imagenet10),
        #("ImageNetDog", load_imagenet_dog)
    ]
    return datasets

"""
def _get_evaluation_algorithms(n_clustering_epochs, embedding_size, batch_size, optimizer_class, loss_fn,
                               augmentation=False):
    evaluation_algorithms = [
        EvaluationAlgorithm("ACeDeC", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),
        EvaluationAlgorithm("AE+KMeans", AEKmeans,
                            {"n_clusters": None, "batch_size": batch_size}),
        EvaluationAlgorithm("DEC", DEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),
        EvaluationAlgorithm("IDEC", IDEC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),
        EvaluationAlgorithm("DCN", DCN,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "clustering_optimizer_params": {"lr": 1e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),
        EvaluationAlgorithm("My_ACeDeC", My_ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),

    ]
    return evaluation_algorithms
"""

def _get_evaluation_algorithms(n_clustering_epochs, embedding_size, batch_size, optimizer_class, loss_fn,
                               augmentation=False):
    evaluation_algorithms = [
        EvaluationAlgorithm("ACeDeC", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),
        EvaluationAlgorithm("My_ACeDeC", My_ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation}),

    ]
    return evaluation_algorithms


def _get_evaluation_metrics():
    evaluation_metrics = [
        EvaluationMetric("NMI", nmi),
        EvaluationMetric("AMI", ami),
        EvaluationMetric("ARI", ari),
        EvaluationMetric("ACC", acc),
        EvaluationMetric("DOM", dom),
    ]
    return evaluation_metrics


if __name__ == "__main__":
    """
    experiment_feedforward_512_256_128_10(datasets=_get_dataset_loaders(),
                                          algorithms_function=_get_evaluation_algorithms,
                                          metrics=_get_evaluation_metrics(),
                                          save_dir=SAVE_DIR, download_path=DOWNLOAD_PATH)
    """
    # experiment_feedforward_500_500_2000_10()
    # experiment_feedforward_500_500_2000_10_aug()
    # experiment_embedding_size()
    experiment_convolutional_resnet18(datasets=_get_dataset_loaders(),
                                          algorithms_function=_get_evaluation_algorithms,
                                          metrics=_get_evaluation_metrics(),
                                          save_dir=SAVE_DIR, download_path=DOWNLOAD_PATH)
    # experiment_convolutional_resnet18_aug()
    # experiment_initial_clustering(save_dir=SAVE_DIR)
    # experiment_learning_rate()
    experiment_feedforward_test_train(datasets=_get_dataset_loaders(),
                                          algorithms_function=_get_evaluation_algorithms,
                                          metrics=_get_evaluation_metrics(),
                                          save_dir=SAVE_DIR, download_path=DOWNLOAD_PATH)