# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from .models import AVAILABLE_MODEL_STRINGS


def add_hyperparameters_to_argparser(parser):
    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=0.0005,
                        help="(Base) learning rate, default: 0.0005")

    parser.add_argument('-m', '--momentum',
                        type=float,
                        default=0.9,
                        help="Momentum to use, default: 0.9")

    parser.add_argument('-ne', '--n_epochs',
                        type=int,
                        default=200,
                        help="Number of epochs to train, default: 200")

    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=128,
                        help="Batch size to use, default: 128")

    parser.add_argument('-opt', '--optimizer',
                        type=str,
                        choices=['sgd', 'adam'],
                        default='adam',
                        help='Optimizer to use, default: adam')

    parser.add_argument('-model', '--model',
                        type=str,
                        choices=AVAILABLE_MODEL_STRINGS,
                        default='mobilenetv2_large_pretrained_additional_dense',
                        help='Network to use, default: '
                             'mobilenetv2_large_pretrained_additional_dense')

    parser.add_argument('-nw', '--num_workers',
                        type=int,
                        default=2,
                        help='Number of Data Workers, default: 2')

    parser.add_argument('-tn', '--training_name',
                        type=str,
                        default='training',
                        help='Name of the training')

    parser.add_argument('-lrd', '--learning_rate_decay',
                        type=float,
                        default=0.9,
                        help='Power of the learning rate poly decay, '
                             'default: 0.9')

    parser.add_argument('-k', '--kappa',
                        type=float,
                        default=1.0,
                        help="Kappa to use for VonMises Loss")

    parser.add_argument('-wm', '--weight_mode',
                        type=str,
                        default='const',
                        choices=['const', 'dwa'],
                        help="mode to use for loss weighting")

    parser.add_argument('-wmdwat', '--weight_mode_dwa_temperature',
                        type=float,
                        default=2,
                        help="temperature to use when weight mode is dwa")

    parser.add_argument('-wmdwam', '--weight_mode_dwa_momentum',
                        type=float,
                        default=0.5,
                        help="momentum to use when weight mode is dwa")

    parser.add_argument('-wd', '--weight_detection',
                        type=float,
                        default=0.1,
                        help="Weight for the Detection Loss")

    parser.add_argument('-wo', '--weight_orientation',
                        type=float,
                        default=0.72,
                        help="Weight for the Orientation Loss")

    parser.add_argument('-wp', '--weight_pose',
                        type=float,
                        default=0.18,
                        help="Weight for the Pose Loss")

    parser.add_argument('-aug', '--augmentation',
                        type=str,
                        default="flip",
                        choices=['flip', 'none'],
                        help="Use augmentation (flip) or no "
                             "augmentation (none)")

    parser.add_argument('-t', '--tasks',
                        type=str,
                        default="detection+orientation+pose",
                        help="Which tasks to train, seperated with +")

    parser.add_argument('-dc', '--dataset_combination',
                        type=str,
                        default='concat',
                        choices=['concat', '50_50'],
                        help='How to combine the datasets. '
                             'Either 50_50 or concat')

    return parser
