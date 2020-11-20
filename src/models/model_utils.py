# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np
from torch import nn


class Identity(nn.Module):
    def __init__(self):
        """
        Used to remove the fc layer after the global average pooling from
        pretrained models.
        """
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def cnt_parameters(model):
    n = 0
    n_trainable = 0
    for param in model.parameters():
        n_param = np.prod(param.size())
        n += n_param
        if param.requires_grad:
            n_trainable += n_param
    return {'total': n, 'trainable': n_trainable}
