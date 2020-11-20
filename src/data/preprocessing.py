# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import numpy as np
from torchvision import transforms

from .img_utils import resize


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        image = resize(image, self.output_size,
                       interpolation=_DEPTH_INTERPOLATION)
        return image[None, ...]   # add channel axis


class ScaleZeroOne(object):
    def __init__(self, scale_factor):
        # self.scale_factor = np.array(scale_factor, dtype=np.float32)
        self.scale_factor = scale_factor

    def __call__(self, image):
        image = image.astype(np.float32)
        image /= self.scale_factor

        return image


class ToFloat(object):
    def __init__(self):
        return

    def __call__(self, image):
        image = image.astype(np.float32)
        return image


class Triple(object):
    """
    Convert a one channel image to a three channel image by creating three
    identical dimensions
    """
    def __call__(self, image):
        return np.tile(image, (3, 1, 1))  # Triple along dim axis


class OrientationAugmentation(object):
    def __call__(self, sample):
        # image, orientation = sample['image'], sample['orientation']

        if np.random.uniform() > 0.5:
            # flip image horizontally
            sample['image'] = sample['image'][:, :, ::-1].copy()
            # flip angle
            sample['orientation'] = (360. - sample['orientation']) % 360.


# default interpolation for depth, rgb and mask images
_DEPTH_INTERPOLATION = 'nearest'
_RGB_INTERPOLATION = 'nearest'
_MASK_INTERPOLATION = 'nearest'


def resize_mask(mask, shape_or_scale):
    return resize(mask, shape_or_scale,
                  interpolation=_MASK_INTERPOLATION)


def resize_depth_img(depth_img, shape_or_scale):
    return resize(depth_img, shape_or_scale,
                  interpolation=_DEPTH_INTERPOLATION)


def resize_rgb_img(rgb_img, shape_or_scale):
    return resize(rgb_img, shape_or_scale,
                  interpolation=_RGB_INTERPOLATION)


def mask_img(img, mask, fill_value=0):
    img[np.logical_not(mask), ...] = fill_value
    return img


def get_preprocessing(model_name):
    preprocessing_ops = []
    if model_name in ['donet_depth_126_48',
                      'donet_depth_126_48_hyperface',
                      'donet_depth_126_48_ms']:
        network_input_size = (126, 48)
    elif model_name == 'efficientnet_b1':
        network_input_size = (240, 240)
    else:
        # all other imagenet pretrained models use 224x224
        network_input_size = (224, 224)

    # triple input if model is a pretrained imagenet model
    if model_name not in ['donet_depth_126_48',
                          'donet_depth_126_48_hyperface',
                          'donet_depth_126_48_ms']:
        network_input_channels = 3
    else:
        network_input_channels = 1

    preprocessing_ops.append(Rescale(network_input_size))
    preprocessing_ops.append(ScaleZeroOne(18000))
    if network_input_channels == 3:
        preprocessing_ops.append(Triple())

    return (transforms.Compose(preprocessing_ops),
            network_input_size,
            network_input_channels)
