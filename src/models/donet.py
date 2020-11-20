# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-05
BN_MOMENTUM = 0.9  # too large?


class DONetClassificationOrientationPose(nn.Module):
    def __init__(self):
        super(DONetClassificationOrientationPose, self).__init__()
        self.activation = F.relu

        self.input_depth_1_conv = nn.Conv2d(1, 24, kernel_size=(3, 3),
                                            stride=(1, 1), padding=1,
                                            bias=False, padding_mode='zeros')
        self.input_depth_1_bn = nn.BatchNorm2d(24, eps=BN_EPS,
                                               momentum=BN_MOMENTUM,
                                               affine=True,
                                               track_running_stats=True)

        self.input_depth_2_conv = nn.Conv2d(24, 24, kernel_size=(3, 3),
                                            stride=(1, 1), padding=1,
                                            bias=False, padding_mode='zeros')
        self.input_depth_2_bn = nn.BatchNorm2d(24, eps=BN_EPS,
                                               momentum=BN_MOMENTUM,
                                               affine=True,
                                               track_running_stats=True)

        self.input_depth_3_conv = nn.Conv2d(24, 24, kernel_size=(3, 3),
                                            stride=(1, 1), padding=1,
                                            bias=False, padding_mode='zeros')
        self.input_depth_3_bn = nn.BatchNorm2d(24, eps=BN_EPS,
                                               momentum=BN_MOMENTUM,
                                               affine=True,
                                               track_running_stats=True)

        self.main_1_pool = nn.MaxPool2d(kernel_size=(3, 2), stride=(3, 2))

        self.main_2_conv = nn.Conv2d(24, 48, kernel_size=(3, 3), stride=(1, 1),
                                     padding=1,
                                     bias=False, padding_mode='zeros')
        self.main_2_bn = nn.BatchNorm2d(48, eps=BN_EPS, momentum=BN_MOMENTUM,
                                        affine=True, track_running_stats=True)

        self.main_3_conv = nn.Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1),
                                     padding=1,
                                     bias=False, padding_mode='zeros')
        self.main_3_bn = nn.BatchNorm2d(48, eps=BN_EPS, momentum=BN_MOMENTUM,
                                        affine=True, track_running_stats=True)

        self.main_4_conv = nn.Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1),
                                     padding=1,
                                     bias=False, padding_mode='zeros')
        self.main_4_bn = nn.BatchNorm2d(48, eps=BN_EPS, momentum=BN_MOMENTUM,
                                        affine=True, track_running_stats=True)

        self.main_5_pool = nn.MaxPool2d(kernel_size=(3, 2), stride=(3, 2))

        self.main_6_conv = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1),
                                     padding=0,
                                     bias=False, padding_mode='zeros')
        self.main_6_bn = nn.BatchNorm2d(64, eps=BN_EPS, momentum=BN_MOMENTUM,
                                        affine=True, track_running_stats=True)

        self.main_7_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
                                     padding=0,
                                     bias=False, padding_mode='zeros')
        self.main_7_bn = nn.BatchNorm2d(64, eps=BN_EPS, momentum=BN_MOMENTUM,
                                        affine=True, track_running_stats=True)

        self.main_8_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.output_2_dropout = nn.Dropout(p=0.2)
        self.output_2_dense = nn.Linear(1280, 512, bias=True)

        self.output_3_dropout = nn.Dropout(p=0.5)
        self.output_3_dense = nn.Linear(512, 7, bias=True)

        # self.output_4_softmax = nn.Softmax()

    def forward(self, x):
        x = self.activation(self.input_depth_1_bn(self.input_depth_1_conv(x)))
        x = self.activation(self.input_depth_2_bn(self.input_depth_2_conv(x)))
        x = self.activation(self.input_depth_3_bn(self.input_depth_3_conv(x)))
        x = self.main_1_pool(x)
        x = self.activation(self.main_2_bn(self.main_2_conv(x)))
        x = self.activation(self.main_3_bn(self.main_3_conv(x)))
        x = self.activation(self.main_4_bn(self.main_4_conv(x)))
        x = self.main_5_pool(x)
        x = self.activation(self.main_6_bn(self.main_6_conv(x)))
        x = self.activation(self.main_7_bn(self.main_7_conv(x)))
        x = self.main_8_pool(x)

        x = x.view(-1, 1280)

        x = self.output_2_dropout(x)
        x = self.output_2_dense(x)
        x = self.activation(x)
        x = self.output_3_dropout(x)
        x = self.output_3_dense(x)
        # Softmax Activation -> Already in CrossEntropyLoss?
        # x = self.output_4_softmax(x)
        return x


def _test():
    import numpy as np
    from torchsummary import summary

    input_shape = (1, 126, 48)
    x = np.random.random((1,)+input_shape).astype('float32')

    model = DONetClassificationOrientationPose()

    # print summary
    print("DONet")
    summary(model, device='cpu', input_size=input_shape)

    # test model
    y = model(torch.tensor(x))
    assert y.detach().numpy().shape == (1, 7)

    dummy_loss = torch.sum(y)
    dummy_loss.backward()
