# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchcv.model_provider import get_model as ptcv_get_model

from .model_utils import Identity


class MobileNetV2ClassificationOrientationPose(nn.Module):
    def __init__(self,
                 width_scale=1.0,
                 pretrained=False,
                 freeze_all_but_last_layer=False,
                 additional_dense=False):
        super(MobileNetV2ClassificationOrientationPose, self).__init__()

        self.additional_dense = additional_dense

        # load mobilenetv2 backbone from pytorchcv
        assert width_scale in [1.0, 0.75, 0.5, 0.25]
        if width_scale == 1.0:
            self.backbone = ptcv_get_model('mobilenetv2_w1',
                                           pretrained=pretrained)
        elif width_scale == 0.75:
            self.backbone = ptcv_get_model('mobilenetv2_w3d4',
                                           pretrained=pretrained)
        elif width_scale == 0.5:
            self.backbone = ptcv_get_model('mobilenetv2_wd2',
                                           pretrained=pretrained)
        else:
            self.backbone = ptcv_get_model('mobilenetv2_wd4',
                                           pretrained=pretrained)

        # freeze parameters
        # note that parameters of subsequently constructed modules have
        # requires_grad=True by default
        if freeze_all_but_last_layer:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # parameters of additional modules have requires_grad=True by default
        n_channels = self.backbone.output.in_channels
        self.n_channels_avg_pool = n_channels

        # replace final conv with identity in order to remove it from the model
        self.backbone.output = Identity()

        # additional dropout after global average pooling
        self.feature_dropout = nn.Dropout(p=0.2)

        if additional_dense:
            # use relu instead of relu6 as in mobilenet v2 backbone
            self.activation = F.relu

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
        x_mobilenetv2 = self.backbone.features(x)
        x_mobilenetv2 = self.backbone.output(x_mobilenetv2)
        # flatten
        x_mobilenetv2 = x_mobilenetv2.view(-1, self.n_channels_avg_pool)
        x_mobilenetv2 = self.feature_dropout(x_mobilenetv2)

        if self.additional_dense:
            output_detection = self.detection_fc_output(x_mobilenetv2)

            x = self.orientation_fc(x_mobilenetv2)
            x = self.activation(x)
            x = self.orientation_dropout_output(x)
            output_orientation = self.orientation_fc_output(x)

            output_pose = self.pose_fc_output(x_mobilenetv2)

            # concat along channel axis to get the same output shape
            return torch.cat([output_detection,
                              output_orientation,
                              output_pose], dim=1)

        else:
            return self.fc_output(x_mobilenetv2)


def _test():
    from itertools import product

    import numpy as np
    from torchsummary import summary

    width_scale = [1.0, 0.75, 0.5, 0.25]
    pretrained = [False, True]
    additional_dense = [False, True]

    input_shape = (3, 224, 224)
    x = np.random.random((1,)+input_shape).astype('float32')

    for ws, pre, add in product(width_scale, pretrained, additional_dense):
        # create model
        model = MobileNetV2ClassificationOrientationPose(
            width_scale=ws,
            pretrained=pre,
            additional_dense=add
        )

        # print summary
        print(f"MobileNetV2 (scale: {ws}, pretrained: {pre}, "
              f"additional_dense: {add})")
        summary(model, device='cpu', input_size=input_shape)

        # test model
        y = model(torch.tensor(x))
        assert y.detach().numpy().shape == (1, 7)

        dummy_loss = torch.sum(y)
        dummy_loss.backward()
