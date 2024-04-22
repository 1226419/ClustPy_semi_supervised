import torch
from clustpy.data import load_mnist
from clustpy.deep import ACeDeC
from clustpy.utils import EvaluationAlgorithm, EvaluationMetric

from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from clustpy.metrics import unsupervised_clustering_accuracy as acc

from clustpy_experiments.experiment_baseline import experiment_baseline
from clustpy.deep.semisupervised_enrc.semi_supervised_acedec import ACeDeC as My_ACeDeC

DOWNLOAD_PATH = "Downloaded_datasets"
SAVE_DIR = "MyBenchmark_Baseline/"


def _get_dataset_loaders():
    datasets = [
        ("MNIST", load_mnist)
    ]
    return datasets


def _get_evaluation_algorithms(n_clustering_epochs, embedding_size, batch_size, optimizer_class, loss_fn,
                               augmentation=False):

    scheduler = torch.optim.lr_scheduler.StepLR
    scheduler_params = {"step_size": int(0.2 * n_clustering_epochs), "gamma": 0.5, "verbose": True}

    evaluation_algorithms = [
        EvaluationAlgorithm("ACeDeC", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "scheduler": scheduler,
                             "scheduler_params": scheduler_params,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation,
                             "final_reclustering": True}),

        EvaluationAlgorithm("My_ACeDeC", My_ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "scheduler": scheduler,
                             "scheduler_params": scheduler_params,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation,
                             "final_reclustering": True}),

    ]
    return evaluation_algorithms


def _get_evaluation_metrics():
    evaluation_metrics = [
        EvaluationMetric("NMI", nmi),
        EvaluationMetric("ARI", ari),
        EvaluationMetric("ACC", acc),
    ]
    return evaluation_metrics


if __name__ == "__main__":
    experiment_baseline(datasets=_get_dataset_loaders(), algorithms_function=_get_evaluation_algorithms,
                        metrics=_get_evaluation_metrics(), save_dir=SAVE_DIR, download_path=DOWNLOAD_PATH)



