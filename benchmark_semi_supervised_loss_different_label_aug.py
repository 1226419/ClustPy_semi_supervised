import torch
from clustpy.data import load_mnist
from clustpy.deep import ACeDeC
from clustpy.utils import EvaluationAlgorithm, EvaluationMetric
from clustpy.metrics import unsupervised_clustering_accuracy as acc
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_init import (semi_supervised_acedec_init,
                                                                        semi_supervised_acedec_init_simple,
                                                                        semi_supervised_acedec_init_simplest)

from clustpy_experiments.experiment_semisupervised_acedec_init import experiment_semi_init_small_ff_multiple_label_percent_aug
from clustpy.deep.semisupervised_enrc.semi_supervised_acedec import ACeDeC as My_ACeDeC
from clustpy.deep.semisupervised_enrc.semi_supervised_enrc_module import _ENRC_Module
from clustpy.deep.semisupervised_enrc.semi_supervised_clustering_modules import _Label_Loss_Module_based_on_ENRC
from clustpy.deep.semisupervised_enrc.semi_supervised_fitting_procedures import enrc_fitting_with_labels

import os



def _get_dataset_loaders():
    datasets = [
        ("MNIST", load_mnist)
    ]
    return datasets


def _get_evaluation_algorithms(n_clustering_epochs, embedding_size, batch_size, optimizer_class, loss_fn,
                               augmentation=True):

    clustering_module = _Label_Loss_Module_based_on_ENRC
    fit_function = enrc_fitting_with_labels
    scheduler = torch.optim.lr_scheduler.StepLR
    scheduler_params = {"step_size": int(0.2 * n_clustering_epochs), "gamma": 0.5, "verbose": True}
    init_kwargs = {"clustering_module": _ENRC_Module, "optimizer_params": {"lr": 1e-3}}
    evaluation_algorithms = [
        EvaluationAlgorithm("SemiACeDeCwithoutsimpleinit", My_ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "scheduler": scheduler,
                             "scheduler_params": scheduler_params, "init_kwargs": init_kwargs,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation,
                             "final_reclustering": True, "clustering_module": clustering_module,
                             "fit_function": fit_function}),
        EvaluationAlgorithm("ACeDeC", ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "scheduler": scheduler,
                             "scheduler_params": scheduler_params,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation,
                             "final_reclustering": True}),

        EvaluationAlgorithm("SemiACeDeC", My_ACeDeC,
                            {"n_clusters": None, "batch_size": batch_size, "clustering_epochs": n_clustering_epochs,
                             "optimizer_class": optimizer_class,
                             "init_subsample_size": 10000,
                             "clustering_optimizer_params": {"lr": 5e-4}, "loss_fn": loss_fn,
                             "scheduler": scheduler,
                             "scheduler_params": scheduler_params,
                             "embedding_size": embedding_size, "augmentation_invariance": augmentation,
                             "init": semi_supervised_acedec_init_simplest, "init_kwargs": init_kwargs,
                             "final_reclustering": True, "clustering_module": clustering_module,
                             "fit_function": fit_function,
                             "reclustering_strategy": "acedec"}, ),
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
    DOWNLOAD_PATH = os.path.join(os.getcwd(), "clustpy_experiments_runs", "Autoencoders", "Downloaded_datasets/")
    SAVE_DIR = os.path.join(os.getcwd(), "clustpy_experiments_runs", "results", "SSVACeDeC_loss_train_test_diff_label_aug/")
    print(SAVE_DIR)
    print(DOWNLOAD_PATH)
    experiment_semi_init_small_ff_multiple_label_percent_aug(datasets=_get_dataset_loaders(), algorithms_function=_get_evaluation_algorithms,
                        metrics=_get_evaluation_metrics(), save_dir=SAVE_DIR, download_path=DOWNLOAD_PATH)



