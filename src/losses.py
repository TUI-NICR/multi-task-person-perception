# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import copy

import numpy as np
import torch


class VonmisesLossBiternion(torch.nn.Module):
    """Von mises loss function for biternion inputs

    see: Beyer et al.: Biternion Nets: Continuous Head Pose Regression from
         Discrete Training Labels, GCPR 2015.
    """
    def __init__(self, kappa):
        super(VonmisesLossBiternion, self).__init__()
        self._kappa = kappa

    def forward(self, prediction, target):
        cos_angles = torch.bmm(prediction[..., None].permute(0, 2, 1),
                               target[..., None])
        cos_angles = torch.exp(self._kappa * (cos_angles - 1))
        score = 1 - cos_angles
        return score[:, 0, 0]


def get_new_loss_weights_dwa(cur_weights, loss_history, tasks,
                             scale_to_number_of_tasks=False,
                             momentum=0.5,
                             t=2):
    # see: Liu et al.: End-to-End Multi-Task Learning with Attention, CVPR 2019.
    # extract losses and weights for the tasks to consider
    losses = [loss_history[task] for task in tasks]
    n_tasks = len(tasks)

    if len(losses[0]) < 2:
        # too few elements in loss history
        # -> do not adapt weights and return current weights
        return cur_weights

    loss_changes = np.array([loss[-1] / loss[-2] for loss in losses])

    # apply temperature scaling
    loss_changes = loss_changes / t

    # get new weights
    loss_changes_exp = np.exp(loss_changes)
    new_weight_values = loss_changes_exp / loss_changes_exp.sum()

    if scale_to_number_of_tasks:
        new_weight_values *= n_tasks

    # update weights in dict
    new_weights = copy.deepcopy(cur_weights)
    for i, task in enumerate(tasks):
        new_weights[task] = momentum*cur_weights[task] + (1-momentum) * \
                            new_weight_values[i]
    return new_weights
