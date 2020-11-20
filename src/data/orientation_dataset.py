# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>

"""
from nicr_rgb_d_orientation_data_set import RGBDOrientationDataset
import numpy as np
from torch.utils.data import Dataset

from .img_utils import resize


class RGBDOrientationDatasetPytorch(Dataset, RGBDOrientationDataset):
    """
    Dataset container class.

    Parameters
    ----------
    dataset_basepath : str
        Path to dataset root, e.g. '/datasets/nicr-multi-task-dataset/'.
    set_name : str
        Set to load, should be one of 'train', 'valid' or 'test'.
    transform : pytorch transform (chain)
        Which transformation should be applied before returning a sample
    augmentation : pytorch augmentation (chain)
        Which augmentation should be applied before returning a sample
    """
    def __init__(self, dataset_basepath, set_name,
                 transform=None, augmentation=None):
        RGBDOrientationDataset.__init__(self, dataset_basepath, set_name)

        self._transform = transform
        self._augmentation = augmentation

    def __getitem__(self, index):
        sample = {
            'image': self._samples[index].get_depth_patch(),
            'is_person': True,
            'orientation': self._samples[index].orientation,
            'pose': 0,   # =standing
            'dataset': 'orientation',
            'tape_name': '',
            'image_id': str(self._samples[index].basename),
            'instance_id': '',
            'person_id': str(self._samples[index].person_id),
            'walk_pattern_type': '',
            'category_name': "person-standing-deeporientation"
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
