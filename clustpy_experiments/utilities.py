import os
import numpy as np
import torch

from clustpy.deep import detect_device, get_dataloader
from clustpy.deep.autoencoders import ConvolutionalAutoencoder
from clustpy.data.preprocessing import z_normalization
from clustpy.utils import EvaluationDataset, EvaluationAutoencoder, evaluate_multiple_datasets

import inspect
import torchvision


def _standardize(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    data = (data - mean) / std
    return data


def _add_color_channels_and_resize(data, conv_used, augmentation, dataset_name):
    if conv_used or augmentation:
        if data.ndim != 4:
            data = data.reshape(-1, 1, data.shape[1], data.shape[2])
            if conv_used:
                data = np.tile(data, (1, 3, 1, 1))
        if conv_used:
            if dataset_name in ["ImageNet10", "ImageNetDog"]:
                size = 224
            else:
                size = 32
            data = torchvision.transforms.Resize((size, size))(torch.from_numpy(data).float()).numpy()
    return data


def _get_evaluation_datasets_with_autoencoders(dataset_loaders, ae_layers, experiment_name, n_repetitions, batch_size,
                                               n_pretrain_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                                               ae_class, other_ae_params, device, save_dir, download_path,
                                               augmentation=False, train_test_split=False):
    evaluation_datasets = []
    # Get autoencoders for DC algortihms
    for data_name_orig, data_loader in dataset_loaders:
        data_name_exp = data_name_orig + "_" + experiment_name
        data_loader_params = inspect.getfullargspec(data_loader).args
        flatten = (ae_class != ConvolutionalAutoencoder and not augmentation)
        train_subset = "train" if train_test_split else "all"
        if augmentation:
            assert flatten == False, "If augmentation is used, flatten must be false"
            # Normalization happens within the augmentation dataloader
            data, labels = data_loader(subset=train_subset, return_X_y=True, downloads_path=download_path)
        elif "normalize_channels" in data_loader_params and not train_test_split:
            data, labels = data_loader(subset=train_subset, return_X_y=True, downloads_path=download_path,
                                       )
            data = z_normalization(data)
        else:
            data, labels = data_loader(subset=train_subset, return_X_y=True, downloads_path=download_path)
            data_mean = np.mean(data)
            data_std = np.std(data)
            data = _standardize(data, data_mean, data_std)
        # Change data format if conv autoencoder is used
        conv_used = ae_class == ConvolutionalAutoencoder
        data = _add_color_channels_and_resize(data, conv_used, augmentation, data_name_orig)
        # Create dataloaders
        if augmentation:
            flatten = ae_class != ConvolutionalAutoencoder  # Update flatten -> should still happen if AE is not Conv
            if not os.path.isdir(save_dir + "{0}/DLs".format(data_name_exp)):
                os.makedirs(save_dir + "{0}/DLs".format(data_name_exp))
            save_path_dl1_aug = save_dir + "{0}/DLs/dl_aug.pth".format(data_name_exp)
            dataloader, orig_dataloader = _get_dataloader_with_augmentation(data, batch_size, flatten, data_name_orig)
            if not os.path.isfile(save_path_dl1_aug):
                torch.save(dataloader, save_path_dl1_aug)
            save_path_dl1_orig = save_dir + "{0}/DLs/dl_orig.pth".format(data_name_exp)
            if not os.path.isfile(save_path_dl1_orig):
                torch.save(orig_dataloader, save_path_dl1_orig)
            path_custom_dataloaders = (save_path_dl1_aug, save_path_dl1_orig)
        else:
            dataloader = get_dataloader(data, batch_size, shuffle=True)
            path_custom_dataloaders = None
        evaluation_autoencoders = []
        for i in range(n_repetitions):
            # Pretrain and save autoencoder
            save_path_ae = save_dir + "{0}/AEs/ae_{1}.ae".format(data_name_exp, i)
            if ae_class != ConvolutionalAutoencoder:
                layers = [data[0].size] + ae_layers
                ae_params = dict(**other_ae_params, **{"layers": layers})
            else:
                ae_params = dict(**other_ae_params, **{"fc_layers": ae_layers, "input_height": data.shape[-1]})
            if not os.path.isfile(save_path_ae):
                ae = ae_class(**ae_params).to(device)
                ae.fit(n_pretrain_epochs, pretrain_optimizer_params, batch_size, dataloader=dataloader,
                       optimizer_class=optimizer_class,
                       loss_fn=loss_fn, device=device, model_path=save_path_ae)
                print("created autoencoder {0} for dataset {1}".format(i, data_name_exp))
            eval_autoencoder = EvaluationAutoencoder(save_path_ae, ae_class, ae_params, path_custom_dataloaders)
            evaluation_autoencoders.append(eval_autoencoder)
        if augmentation:
            eval_dataset = EvaluationDataset(data_name_exp, data_loader,
                                             data_loader_params={"downloads_path": download_path},
                                             iteration_specific_autoencoders=evaluation_autoencoders,
                                             train_test_split=train_test_split,
                                             preprocess_methods=_add_color_channels_and_resize,
                                             preprocess_params={"conv_used": conv_used, "augmentation": augmentation,
                                                                "dataset_name": data_name_orig})
        elif "normalize_channels" in data_loader_params and not train_test_split:
            eval_dataset = EvaluationDataset(data_name_exp, data_loader,
                                             data_loader_params={"normalize_channels": True,
                                                                 "downloads_path": download_path},
                                             iteration_specific_autoencoders=evaluation_autoencoders,
                                             train_test_split=train_test_split,
                                             preprocess_methods=_add_color_channels_and_resize,
                                             preprocess_params={"conv_used": conv_used, "augmentation": augmentation,
                                                                "dataset_name": data_name_orig})
        else:
            eval_dataset = EvaluationDataset(data_name_exp, data_loader,
                                             data_loader_params={"downloads_path": download_path},
                                             iteration_specific_autoencoders=evaluation_autoencoders,
                                             train_test_split=train_test_split,
                                             preprocess_methods=[_standardize, _add_color_channels_and_resize],
                                             preprocess_params=[{"mean": data_mean, "std": data_std},
                                                                {"conv_used": conv_used, "augmentation": augmentation,
                                                                 "dataset_name": data_name_orig}])
        evaluation_datasets.append(eval_dataset)
    return evaluation_datasets


def _experiment(experiment_name, ae_layers, embedding_size, n_repetitions, batch_size,
                n_pretrain_epochs, n_clustering_epochs, optimizer_class, pretrain_optimizer_params, loss_fn,
                ae_class, other_ae_params, dataset_loaders, get_evaluation_algorithmns_function, evaluation_metrics,
                save_dir, download_path,
                augmentation=False, train_test_split=False):
    ae_layers = ae_layers.copy()
    ae_layers.append(embedding_size)
    experiment_name = experiment_name + "_" + "_".join(str(x) for x in ae_layers)

    device = detect_device()
    evaluation_datasets = _get_evaluation_datasets_with_autoencoders(dataset_loaders, ae_layers, experiment_name,
                                                                     n_repetitions, batch_size, n_pretrain_epochs,
                                                                     optimizer_class, pretrain_optimizer_params,
                                                                     loss_fn, ae_class, other_ae_params,
                                                                     device, download_path, save_dir,
                                                                     augmentation, train_test_split)
    evaluation_algorithms = get_evaluation_algorithmns_function(n_clustering_epochs, embedding_size, batch_size,
                                                                optimizer_class, loss_fn, augmentation)

    evaluate_multiple_datasets(evaluation_datasets, evaluation_algorithms, evaluation_metrics, n_repetitions,
                               add_runtime=True, add_n_clusters=False,
                               save_path=save_dir + experiment_name + "/Results/result.csv",
                               save_intermediate_results=True,
                               save_labels_path=save_dir + experiment_name + "/Labels/label.csv")


def _get_dataloader_with_augmentation(data: np.ndarray, batch_size: int, flatten: int, data_name: str):
    data = torch.tensor(data)
    data /= 255.0
    channel_means = data.mean([0, 2, 3])
    channel_stds = data.std([0, 2, 3])
    # preprocessing functions
    normalize_fn = torchvision.transforms.Normalize(channel_means, channel_stds)
    flatten_fn = torchvision.transforms.Lambda(torch.flatten)
    # augmentation transforms
    if data_name in ["CIFAR10", "ImageNet10", "ImageNetDog"]:
        # color image augmentation according to SimCLR: https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/data_aug/contrastive_learning_dataset.py#L13
        _size = data.shape[-1]
        radias = int(0.1 * _size) // 2
        kernel_size = radias * 2 + 1
        color_jitter = torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        transform_list = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomResizedCrop(size=_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([color_jitter], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.GaussianBlur(kernel_size=kernel_size),
            torchvision.transforms.ToTensor(),
            normalize_fn,
        ]
    else:
        transform_list = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomAffine(degrees=(-16, +16), translate=(0.1, 0.1), shear=(-8, 8), fill=0),
            torchvision.transforms.ToTensor(),
            normalize_fn
        ]
    orig_transform_list = [normalize_fn]
    if flatten:
        transform_list.append(flatten_fn)
        orig_transform_list.append(flatten_fn)
    aug_transforms = torchvision.transforms.Compose(transform_list)
    orig_transforms = torchvision.transforms.Compose(orig_transform_list)
    # pass transforms to dataloader
    aug_dl = get_dataloader(data, batch_size=batch_size, shuffle=True,
                            ds_kwargs={"aug_transforms_list": [aug_transforms],
                                       "orig_transforms_list": [orig_transforms]})
    orig_dl = get_dataloader(data, batch_size=batch_size, shuffle=False,
                             ds_kwargs={"orig_transforms_list": [orig_transforms]})
    return aug_dl, orig_dl
