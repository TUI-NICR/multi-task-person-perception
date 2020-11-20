# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
import torch.nn as nn

from pytorchcv.model_provider import get_model as ptcv_get_model

from .model_utils import Identity


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EfficientNetClassificationOrientationPose(nn.Module):
    def __init__(self,
                 model='b0',
                 tf_mode=False,    # see notes below
                 pretrained=False,
                 freeze_all_but_last_layer=False,
                 additional_dense=False):
        super(EfficientNetClassificationOrientationPose, self).__init__()

        self.additional_dense = additional_dense

        # load efficientnet backbone from pytorchcv
        # notes:
        # - models are ported from tensorflow to pytorch, which means that
        #   the padding is different when using same padding and
        #   stride > 1 at the same time (top-left vs bottom-right)
        # - if `tf_mode` is True, padding is applied manually and exactly as
        #   in TF, but this also comes at some additional cost.
        # - if `tf_mode` is False, default padding from pytorch is applied,
        #   which leads to a small performance decrease (top1 / top5):
        #   - b0:  23.88 / 7.02 -> 24.77 / 7.52
        #   - b1:  21.60 / 5.94 -> 23.08 / 6.38

        model_identifier = f'efficientnet_{model}{"b" if tf_mode else ""}'
        self.backbone = ptcv_get_model(model_identifier,
                                       pretrained=pretrained)

        # freeze parameters
        # note that parameters of subsequently constructed modules have
        # requires_grad=True by default
        if freeze_all_but_last_layer:
            for param in self.backbone.parameters():
                param.requires_grad = False

        n_channels = self.backbone.output.fc.in_features
        self.n_channels_avg_pool = n_channels

        # replace fully connected layer identity in order to remove it
        self.backbone.output.fc = Identity()

        if additional_dense:
            self.activation = Swish()

            # detection output
            self.detection_fc_output = nn.Linear(n_channels, 2)

            # orientation output including additional fc layer
            self.orientation_fc = nn.Linear(n_channels, 512, bias=True)
            self.orientation_dropout_output = nn.Dropout(p=0.5)
            self.orientation_fc_output = nn.Linear(512, 2, bias=True)

            # pose output
            self.pose_fc_output = nn.Linear(n_channels, 3, bias=True)
        else:
            self.fc_output = nn.Linear(n_channels, 7)

    def forward(self, x):
        # apply backbone step by step to skip view op in original forward
        x_efficientnet = self.backbone.features(x)
        x_efficientnet = x_efficientnet.view(-1, self.n_channels_avg_pool)
        x_efficientnet = self.backbone.output(x_efficientnet)

        if self.additional_dense:
            output_detection = self.detection_fc_output(x_efficientnet)

            x = self.orientation_fc(x_efficientnet)
            x = self.activation(x)
            x = self.orientation_dropout_output(x)
            output_orientation = self.orientation_fc_output(x)

            output_pose = self.pose_fc_output(x_efficientnet)

            # concat along channel axis to get the same output shape
            return torch.cat([output_detection,
                              output_orientation,
                              output_pose], dim=1)

        else:
            return self.fc_output(x_efficientnet)


def _test():
    from itertools import product

    import numpy as np
    from torchsummary import summary

    model = ['b0', 'b1']
    tf_mode = [False, True]
    pretrained = [False, True]
    additional_dense = [False, True]

    for m, tf, pre, add in product(model, tf_mode, pretrained,
                                   additional_dense):
        # create input
        if m == 'b0':
            input_shape = (3, 224, 224)
        elif m == 'b1':
            input_shape = (3, 240, 240)

        x = np.random.random((1,) + input_shape).astype('float32')

        # create model
        model = EfficientNetClassificationOrientationPose(
            model=m,
            tf_mode=tf,
            pretrained=pre,
            additional_dense=add
        )

        # print summary
        print(f"EfficientNet ({m}, tfmode: {tf}, pretrained: {pre}, "
              f"additional_dense: {add})")
        summary(model, device='cpu', input_size=input_shape)

        # test model
        y = model(torch.tensor(x))
        assert y.detach().numpy().shape == (1, 7)

        dummy_loss = torch.sum(y)
        dummy_loss.backward()
