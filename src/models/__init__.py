# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from itertools import product

from .donet import DONetClassificationOrientationPose
from .donet_ms import DONetMultiscaleClassificationOrientationPose
from .efficientnet import EfficientNetClassificationOrientationPose
from .mobilenet_v2 import MobileNetV2ClassificationOrientationPose
from .resnet_resnext import ResNet18ClassificationOrientationPose
from .resnet_resnext import ResNet34ClassificationOrientationPose
from .resnet_resnext import ResNet50ClassificationOrientationPose
from .resnet_resnext import ResNeXt50ClassificationOrientationPose


AVAILABLE_MODEL_STRINGS = [
    'donet_depth_126_48',
    'donet_depth_126_48_ms',
]
AVAILABLE_MODEL_STRINGS += [
    f'{b}{a}{p}'
    for b, a, p in product(['efficientnet_b0', 'efficientnet_b1',
                            'mobilenetv2_small', 'mobilenetv2_large',
                            'resnet18', 'resnet34', 'resnet50',
                            'resnext50'],
                           ['_additional_dense', ''],
                           ['_pretrained', ''])
]


def get_model_by_string(model_string, device, **model_kwargs):
    # derive model class and args from
    model_cls = None

    if model_string == 'donet_depth_126_48':
        model_cls = DONetClassificationOrientationPose
    elif model_string == 'donet_depth_126_48_ms':
        model_cls = DONetMultiscaleClassificationOrientationPose
    else:
        # all other model support the following args
        model_kwargs['pretrained'] = model_string.find('pretrained') > -1
        model_kwargs['additional_dense'] = \
            model_string.find('additional_dense') > -1
        model_kwargs['freeze_all_but_last_layer'] = \
            model_kwargs.get('freeze_all_but_last_layer', False)

        if model_string.startswith('mobilenetv2'):
            model_cls = MobileNetV2ClassificationOrientationPose

            if model_string.startswith('mobilenetv2_small'):
                model_kwargs['width_scale'] = 0.25
            elif model_string.startswith('mobilenetv2_large'):
                model_kwargs['width_scale'] = 1.0

        elif model_string.startswith('efficientnet'):
            model_cls = EfficientNetClassificationOrientationPose

            # use default padding from pytorch, see efficientnet class
            model_kwargs['tf_mode'] = False

            if model_string.startswith('efficientnet_b0'):
                model_kwargs['model'] = 'b0'
            elif model_string.startswith('efficientnet_b1'):
                model_kwargs['model'] = 'b1'

        elif model_string.startswith('resnet'):
            if model_string.startswith('resnet18'):
                model_cls = ResNet18ClassificationOrientationPose
            elif model_string.startswith('resnet34'):
                model_cls = ResNet34ClassificationOrientationPose
            elif model_string.startswith('resnet50'):
                model_cls = ResNet50ClassificationOrientationPose

            # multi-scale
            model_kwargs['multiscale'] = model_string.find('_ms') > -1

        elif model_string.startswith('resnext50'):
            model_cls = ResNeXt50ClassificationOrientationPose

            # multi-scale
            model_kwargs['multiscale'] = model_string.find('_ms') > -1

    if model_cls is None:
        raise ValueError(f"Cannot load {model_string}")

    print(f"Loading model: {model_cls} with args: {model_kwargs}")
    model = model_cls(**model_kwargs)
    return model.to(device=device)


def _test():
    for model_string in AVAILABLE_MODEL_STRINGS:
        print(model_string)
        model = get_model_by_string(model_string, device='cpu')
