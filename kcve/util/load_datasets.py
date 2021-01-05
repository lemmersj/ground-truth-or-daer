"""Loads datasets

Loads a different dataset based on a string argument.
"""
# pylint: disable=W0613, W0614
import os
import torch
from torchvision import transforms
from datasets import pascal_dataset, both_dataset
from .Paths import *

DATASET_ROOT = pascal3d_root

def get_data_loaders(
        dataset, batch_size, num_workers, num_classes=12, return_kp=False,
        classifier=False, filter_threshold=5):
    """Gets a dataloader.

    Args:
        dataset: Which dataset to load. One of both, pascalVehKP.
        batch_size: the batch size.
        num_workers: how many workers the dataloader uses.

    returns:
        A dataloader corresponding to the given dataset parameters.
    """

    image_size = 227
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=(0., 0., 0.),
                                              std=(1./255., 1./255., 1./255.)),
                                          transforms.Normalize(
                                              mean=(104, 116.668, 122.678),
                                              std=(1., 1., 1.))])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=(0., 0., 0.),
                                             std=(1./255., 1./255., 1./255.)),
                                         transforms.Normalize(
                                             mean=(104, 116.668, 122.678),
                                             std=(1., 1., 1.))])

    if dataset == "both":
        syn_train_csv = os.path.join(
            root_dir, 'csv_files/synthetic_random_train.csv')

        real_train_csv = os.path.join(
            root_dir, 'csv_files/veh_pascal3d_kp_train.csv')

        train_set = both_dataset(synthetic_csv=syn_train_csv,
                                 pascal_csv=real_train_csv,
                                 synthetic_root=syn_dataset_root,
                                 pascal_root=DATASET_ROOT,
                                 transform=train_transform,
                                 im_size=image_size)

    if dataset == "pascalVehKP":
        csv_train = os.path.join(
            root_dir, 'csv_files/veh_pascal3d_kp_train.csv')

        train_set = pascal_dataset(csv_train,
                                   dataset_root=DATASET_ROOT,
                                   transform=train_transform,
                                   im_size=image_size,
                                   load_adv=False,
                                   load_rand=True,
                                   augment=True)

    # Generate data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    return train_loader, None
