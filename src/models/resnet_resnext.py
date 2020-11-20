# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from .model_utils import Identity


class _BaseClassificationOrientationPose(nn.Module):
    def __init__(self,
                 backbone,
                 freeze_all_but_last_layer=False,
                 additional_dense=False,
                 multiscale=False):
        super(_BaseClassificationOrientationPose, self).__init__()

        self.additional_dense = additional_dense
        self.multiscale = multiscale
        self.backbone = backbone

        # freeze parameters
        # note that parameters of subsequently constructed modules have
        # requires_grad=True by default
        if freeze_all_but_last_layer:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if not multiscale:
            n_channels = self.backbone.fc.in_features
        else:
            # accumulate number features / channels
            n_channels = 0
            layers = [
                # self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4
            ]
            for l in layers:
                # get last block in module (resnet stage)
                *_, last_block = l.children()
                # get last layer in block
                *_, second_last, last = last_block.children()
                try:
                    # resnet18, resnet34: last = batchnorm
                    n_channels += last.num_features
                except AttributeError:
                    # resnet50, resnext50 last = relu, second_last = batchnorm
                    n_channels += second_last.num_features
        self.n_channels_avg_pool = n_channels

        # replace fully connected layer in backbone
        self.backbone.fc = Identity()

        # additional dropout after global average pooling
        self.feature_dropout = nn.Dropout(p=0.2)

        if additional_dense:
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
        # backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        if not self.multiscale:
            x_backbone = self.backbone.avgpool(x4)
        else:
            # x1_backbone = self.backbone.avgpool(x1)
            x2_backbone = self.backbone.avgpool(x2)
            x3_backbone = self.backbone.avgpool(x3)
            x4_backbone = self.backbone.avgpool(x4)
            x_backbone = torch.cat([
                # x1_backbone,
                x2_backbone,
                x3_backbone,
                x4_backbone
            ], dim=1)

        x_backbone = torch.flatten(x_backbone, 1)
        x_backbone = self.feature_dropout(x_backbone)

        if self.additional_dense:
            output_detection = self.detection_fc_output(x_backbone)

            x = self.orientation_fc(x_backbone)
            x = self.activation(x)
            x = self.orientation_dropout_output(x)
            output_orientation = self.orientation_fc_output(x)

            output_pose = self.pose_fc_output(x_backbone)

            # concat along channel axis to get the same output shape
            return torch.cat([output_detection,
                              output_orientation,
                              output_pose], dim=1)

        else:
            return self.fc_output(x_backbone)


_Base = _BaseClassificationOrientationPose    # pep8 ;)


class ResNet18ClassificationOrientationPose(_Base):
    def __init__(self,
                 pretrained=False,
                 freeze_all_but_last_layer=False,
                 additional_dense=False,
                 multiscale=False):
        backbone = models.resnet18(pretrained=pretrained)
        super(ResNet18ClassificationOrientationPose, self).__init__(
            backbone=backbone,
            freeze_all_but_last_layer=freeze_all_but_last_layer,
            additional_dense=additional_dense,
            multiscale=multiscale
        )


class ResNet34ClassificationOrientationPose(_Base):
    def __init__(self,
                 pretrained=False,
                 freeze_all_but_last_layer=False,
                 additional_dense=False,
                 multiscale=False):
        backbone = models.resnet34(pretrained=pretrained)
        super(ResNet34ClassificationOrientationPose, self).__init__(
            backbone=backbone,
            freeze_all_but_last_layer=freeze_all_but_last_layer,
            additional_dense=additional_dense,
            multiscale=multiscale
        )


class ResNet50ClassificationOrientationPose(_Base):
    def __init__(self,
                 pretrained=False,
                 freeze_all_but_last_layer=False,
                 additional_dense=False,
                 multiscale=False):
        backbone = models.resnet50(pretrained=pretrained)
        super(ResNet50ClassificationOrientationPose, self).__init__(
            backbone=backbone,
            freeze_all_but_last_layer=freeze_all_but_last_layer,
            additional_dense=additional_dense,
            multiscale=multiscale
        )


class ResNeXt50ClassificationOrientationPose(_Base):
    def __init__(self,
                 pretrained=False,
                 freeze_all_but_last_layer=False,
                 additional_dense=False,
                 multiscale=False):
        backbone = models.resnext50_32x4d(pretrained=pretrained)
        super(ResNeXt50ClassificationOrientationPose, self).__init__(
            backbone=backbone,
            freeze_all_but_last_layer=freeze_all_but_last_layer,
            additional_dense=additional_dense,
            multiscale=multiscale
        )


def _test():
    from itertools import product

    import numpy as np
    from torchsummary import summary

    classes = [ResNet18ClassificationOrientationPose,
               ResNet34ClassificationOrientationPose,
               ResNet50ClassificationOrientationPose,
               ResNeXt50ClassificationOrientationPose]
    pretrained = [False, True]
    multiscale = [False, True]
    additional_dense = [False, True]

    input_shape = (3, 224, 224)
    x = np.random.random((1,)+input_shape).astype('float32')

    for cls, pre, add, ms in product(classes, pretrained, additional_dense,
                                     multiscale):
        # create model
        model = cls(pretrained=pre, additional_dense=add, multiscale=ms)

        # print summary
        cls_name = cls.__name__
        print(f"{cls_name} (pretrained: {pre}, additional_dense: {add}, "
              f"ms: {ms})")
        summary(model, device='cpu', input_size=input_shape)

        # test model
        y = model(torch.tensor(x))
        assert y.detach().numpy().shape == (1, 7)

        dummy_loss = torch.sum(y)
        dummy_loss.backward()
