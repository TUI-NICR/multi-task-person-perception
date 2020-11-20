# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>

"""
from nicr_multi_task_dataset import MultiTaskDataset
import numpy as np
from torch.utils.data import Dataset

from .img_utils import resize


class MultiTaskDatasetPytorch(Dataset, MultiTaskDataset):
    """
    Dataset container class.

    Parameters
    ----------
    dataset_basepath : str
        Path to dataset root, e.g. '/datasets/NICR-Multi-Task-Dataset/'.
    set_name : str
        Set to load, should be one of 'train', 'valid' or 'test'.
    transform : pytorch transform (chain)
        Which transformation should be applied before returning a sample
    augmentation : pytorch augmentation (chain)
        Which augmentation should be applied before returning a sample
    """
    def __init__(self, dataset_basepath, set_name,
                 transform=None, augmentation=None):
        MultiTaskDataset.__init__(self, dataset_basepath, set_name)

        self._transform = transform
        self._augmentation = augmentation

    def __getitem__(self, index):
        # return self._samples[index]
        split = self._samples[index].basename.split('_')
        sample = {
            'image': self._samples[index].get_depth_patch(),
            'is_person': self._samples[index].is_person,
            'orientation': self._samples[index].orientation,
            'pose': self._samples[index].posture_class,
            'dataset': 'nicr-multi-task',
            'tape_name': self._samples[index].tape_name,
            'image_id': split[0],
            'instance_id': split[1],
            'person_id': self._samples[index].person_name,
            'walk_pattern_type': '',
            'category_name': self._samples[index].category_name
        }
        if self._transform:
            sample['image'] = self._transform(sample['image'])

            mask = self._samples[index].get_mask_patch()
            mask_resized = resize(mask, (sample['image'].shape[1],
                                         sample['image'].shape[2]),
                                  interpolation='nearest')
            mask_resized = mask_resized > 0
            sample['image'][..., np.logical_not(mask_resized)] = 0

        if self._augmentation:
            self._augmentation(sample)

        return sample
