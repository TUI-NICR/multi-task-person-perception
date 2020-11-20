# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import numpy as np
import torch


def rad2deg(rad):
    return np.rad2deg(rad.flatten()) % 360


def biternion2deg(biternion):
    rad = np.arctan2(biternion[:, 1], biternion[:, 0])
    return rad2deg(rad)


def normalize_orientation_output(x):
    return x / torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
