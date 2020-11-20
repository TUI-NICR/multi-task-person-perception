# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import os
from .nicr_multi_task_dataset import MultiTaskDatasetPytorch
from .orientation_dataset import RGBDOrientationDatasetPytorch


def get_combined_dataset(list_of_dataset_strings,
                         set_name,
                         transform,
                         basepath,
                         augmentation=None):
    datasets = []

    assert set_name in ['train', 'valid', 'test']

    # NICR Multi-Task Dataset
    if 'multitask' in list_of_dataset_strings:
        # NICR Multi-Task Dataset Train
        if set_name == 'train':
            datasets.append(MultiTaskDatasetPytorch(
                os.path.join(basepath, 'NICR-Multi-Task-Dataset'),
                set_name='train',
                transform=transform,
                augmentation=augmentation)
            )

        # NICR Multi-Task Dataset Valid
        elif set_name == 'valid':
            datasets.append(MultiTaskDatasetPytorch(
                os.path.join(basepath, 'NICR-Multi-Task-Dataset'),
                set_name='valid',
                transform=transform,
                augmentation=None)
            )

        # NICR Multi-Task Dataset Test
        elif set_name == 'test':
            datasets.append(MultiTaskDatasetPytorch(
                os.path.join(basepath, 'NICR-Multi-Task-Dataset'),
                set_name='test',
                transform=transform,
                augmentation=None)
            )

    # NICR Orientation Dataset
    if 'orientation' in list_of_dataset_strings:
        # NICR Orientation Dataset Train -----------------
        if set_name == 'train':
            datasets.append(RGBDOrientationDatasetPytorch(
                os.path.join(basepath, 'NICR-RGB-D-Orientation-Data-Set'),
                set_name='training',
                transform=transform,
                augmentation=augmentation)
            )

        # NICR Orientation Dataset Valid
        elif set_name == 'valid':
            datasets.append(RGBDOrientationDatasetPytorch(
                os.path.join(basepath, 'NICR-RGB-D-Orientation-Data-Set'),
                set_name='validation',
                transform=transform,
                augmentation=None)
            )

        # NICR Orientation Dataset Test
        elif set_name == 'test':
            datasets.append(RGBDOrientationDatasetPytorch(
                os.path.join(basepath, 'NICR-RGB-D-Orientation-Data-Set'),
                set_name='test',
                transform=transform,
                augmentation=None)
            )

    return datasets
