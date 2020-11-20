#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.postprocessing import biternion2deg
from src.data.postprocessing import normalize_orientation_output
from src.data.preprocessing import get_preprocessing
from src.models import get_model_by_string
from src.data import img_utils


def _parse_args():
    """Parse command-line arguments"""
    parser = ap.ArgumentParser(
        formatter_class=ap.RawTextHelpFormatter,
        description='Test already trained neural network on sample images'
    )
    # evaluation (test)
    parser.add_argument('-m', '--model_path',
                        type=str,
                        default='./trained_networks/mobilenetv2_large_1.json',
                        help='Path to .json file for the model to use')

    parser.add_argument('-i', '--images_path',
                        type=str,
                        default='./samples',
                        help='Path to the folder to load images from')

    return parser.parse_args()


def main():
    args = _parse_args()

    # use CUDA if possible
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # get the network weights filepath, network type and person threshold
    with open(args.model_path) as file:
        network_info = json.load(file)
        weight_path = os.path.join(os.path.dirname(args.model_path),
                                   network_info['weights_file'])
        detection_threshold = network_info['detection_validation_threshold']
        network_type = network_info['network_type']

    # get transformations for preprocessing
    data_transform, network_input_size, network_input_channels = \
        get_preprocessing(network_type)

    # load samples
    filenames = sorted(os.listdir(args.images_path))
    filenames = list(filter(lambda x: os.path.splitext(x)[1] == '.pgm',
                            filenames))
    n_images = len(filenames)
    batched_data = np.empty((n_images,
                             network_input_channels,
                             network_input_size[0],
                             network_input_size[1]), np.float32)
    images = []
    for i, fn in enumerate(filenames):
        image = img_utils.load(os.path.join(args.images_path, fn))
        images.append(image)
        image = data_transform(image)

        batched_data[i] = image

    # create model on GPU / CPU
    model = get_model_by_string(network_type, device)
    # load weights
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # network inference
    model.eval()
    softmax = torch.nn.Softmax(dim=1).to(device=device)
    with torch.set_grad_enabled(False):
        x = torch.tensor(batched_data, device=device)
        output = model(x)

        detection_output = softmax(output[:, 0:2]).cpu().numpy()
        orientation_output = \
            normalize_orientation_output(output[:, 2:4]).cpu().numpy()
        posture_output = softmax(output[:, 4:7]).cpu().numpy()

    results = {
        'detection': detection_output[:, 1] > detection_threshold,
        'orientation': biternion2deg(orientation_output),
        'posture': np.argmax(posture_output, axis=1)
    }

    # plot results
    plot_width = 4
    plot_height = int(np.ceil(n_images / plot_width))

    fig, axes = plt.subplots(plot_width, plot_height,
                             figsize=(6, 6),
                             constrained_layout=True)
    for i, (image, fn) in enumerate(zip(images, filenames)):
        pos_y = i % plot_width
        pos_x = int(np.floor(i / plot_width))

        ax = axes[pos_x, pos_y]
        ax.imshow(image,
                  cmap='gray',
                  vmin=image[image > 0].min(),    # ignore zero for cmap scaling
                  vmax=image.max())
        if results['detection'][i]:    # person
            posture = results['posture'][i]
            if posture == 0:
                posture = 'standing'
                orientation = results['orientation'][i]
                orientation_string = f', {orientation:0.1f}°'
            elif posture == 1:
                posture = 'squatting'
                orientation_string = ''
            else:
                posture = 'sitting'
                orientation_string = ''
            ax.set_title(f'{fn}\nPerson, {posture}{orientation_string}',
                         fontsize=8)
        else:
            ax.set_title(f'{fn}\nNon Person', fontsize=8)
        ax.tick_params(axis='both', labelsize=6)

    fig.suptitle(f'Model: {args.model_path}', fontsize=10)
    plt.savefig('./img/results_samples.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
