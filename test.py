#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import json
import os
import pickle
import shlex

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import sklearn.metrics as metrics
import torch
from tqdm import tqdm

from src.data.dataset import get_combined_dataset
from src.data.postprocessing import biternion2deg
from src.data.postprocessing import normalize_orientation_output
from src.data.preprocessing import get_preprocessing
from src.evaluation_utils import get_statistics_binary
from src.evaluation_utils import pr_measures
from src.evaluation_utils import roc_measures
from src.models import get_model_by_string
from src.parameters import add_hyperparameters_to_argparser
from src.io_utils import create_directory_if_not_exists


def _parse_args():
    """Parse command-line arguments"""
    desc = 'Test trained neural network for multi-task person perception'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)
    # evaluation (test)
    parser.add_argument('-e', '--evaluation_path',
                        type=str,
                        default='/resnet18_coolModel',
                        help='Path to evaluation folder')
    parser.add_argument('-cbeb', '--choose_best_epoch_by',
                        type=str,
                        default="ranked_sum",
                        choices=["detection", "orientation", "pose",
                                 "ranked_sum"],
                        help='How to choose the best epoch, (by orientation, '
                             'detection, pose, or some combined measure)')
    parser.add_argument('-bs', '--batch_size',
                        type=int,
                        default=128,
                        help="Batch Size to use for evaluation")

    parser.add_argument('-dm', '--detection_measure',
                        type=str,
                        default='f1',
                        choices=['f1', 'balanced_accuracy'],
                        help='Measure to use for determining the best epoch '
                             'for the detection task')

    # dataset -----------------------------------------------------------------
    parser.add_argument('-db', '--dataset_basepath',
                        type=str,
                        default='./datasets',
                        help='Path to downloaded dataset')

    parser.add_argument('-ds', '--datasets',
                        type=str,
                        default='multitask+orientation',
                        help='Datasets to use, seperated by +, '
                             'e.g. multitask+orientation or multi-task')

    return parser.parse_args()


def parse_train_and_valid_results(filepath, choose_type, detection_measure):
    # ensure that detection_measure is one of the measures below (max -> best)
    assert detection_measure in ['f1', 'balanced_accuracy']

    # load arguments
    with open(os.path.join(filepath, 'argument_list.txt')) as file:
        parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
        add_hyperparameters_to_argparser(parser)
        args, _ = parser.parse_known_args(shlex.split(file.read()))

    tasks = args.tasks.split('+')
    tasks.sort()

    # read csv file
    df = pd.read_csv(os.path.join(filepath, 'training.csv'),
                     index_col=False)
    training = df.to_dict('list')

    if choose_type == 'detection':
        best_epoch = np.argmax(training[f'detection_valid_{detection_measure}'])
    elif choose_type == 'orientation':
        best_epoch = np.argmin(training['orientation_valid_mae'])
    elif choose_type == 'pose':
        best_epoch = np.argmax(training['pose_valid_balanced_accuracy'])
    elif choose_type == 'ranked_sum':
        ranking = np.zeros_like(training['epoch'])
        if f'detection_valid_{detection_measure}' in training:
            values = np.array(training[f'detection_valid_{detection_measure}'])
            ranking += rankdata(values.max()-values, method='min')
        if 'orientation_valid_mae' in training:
            ranking += rankdata(training['orientation_valid_mae'],
                                method='min')
        if 'pose_valid_balanced_accuracy' in training:
            values = np.array(training['pose_valid_balanced_accuracy'])
            ranking += rankdata(values.max()-values, method='min')

        best_epoch = np.argmin(ranking)
    else:
        raise ValueError("Network evaluation by choosing best network by:"
                         f"{choose_type} is not supported.")

    print(f'Best Epoch: {best_epoch}')
    results = {
        'tasks': tasks,
        'chosen_network_by': choose_type,
        'epoch': best_epoch,
        'weights_filepath': os.path.join(filepath,
                                         'models',
                                         f'epoch_{best_epoch}.pt'),
        'model': args.model,
        'training_name': os.path.basename(filepath),
    }

    # append all the validation results from the CSV file to the result dict
    if 'detection' in tasks:
        results[f'detection_valid_{detection_measure}'] = \
            training[f'detection_valid_{detection_measure}'][best_epoch]
        results[f'detection_valid_{detection_measure}_thresh'] = \
            training[f'detection_valid_{detection_measure}_thresh'][best_epoch]

        print("Detection Valid Results:"
              f"\n\tepoch:\t\t{best_epoch}"
              f"\n\t{detection_measure}:"
              f"\t\t{results[f'detection_valid_{detection_measure}']}"
              f"\n\tthreshold:"
              f"\t{results[f'detection_valid_{detection_measure}_thresh']}")

    if 'pose' in tasks:
        results['pose_valid_balanced_accuracy'] = \
            training['pose_valid_balanced_accuracy'][best_epoch]

        print("Pose Valid Results:"
              f"\n\tepoch:\t\t{best_epoch}"
              f"\n\tbacc:\t\t{results['pose_valid_balanced_accuracy']}")

    if 'orientation' in tasks:
        results['valid_orientation_mae'] = \
            training['orientation_valid_mae'][best_epoch]

        print(f"Orientation Valid Results for:"
              f"\n\tepoch:\t\t{best_epoch}"
              f"\n\tmae:\t\t{results['valid_orientation_mae']}")

    print('\n')

    # create .json file for further use without needing the validation data:
    json_out = {
        'weights_file': f'models/epoch_{best_epoch}.pt',
        'detection_validation_threshold': results[f'detection_valid_{detection_measure}_thresh'],
        'network_type': results['model'],
        'tasks': results['tasks']
    }
    with open(os.path.join(filepath, 'model.json'), 'w') as f:
        json.dump(json_out, f)

    return results


def main():
    args = _parse_args()
    dataset_names = args.datasets.split('+')
    # sort datasets, so that we always have the same order for the means
    dataset_names.sort()

    # depending on evaluation_path, we want to evaluate a single model (if a
    # path to a .json is given) or we want to evaluate a training series (if a
    # directory name is given)
    if args.evaluation_path.endswith('.json'):
        # just use the given model
        with open(args.evaluation_path) as f:
            network_info = json.load(f)

        trainings_settings_and_results = {  # Mapping the names
            'model': network_info['network_type'],
            'tasks': network_info['tasks'],
            f'detection_valid_{args.detection_measure}_thresh': network_info['detection_validation_threshold'],
            'weights_filepath': os.path.join(os.path.dirname(args.evaluation_path), network_info['weights_file']),
            'epoch': np.nan,  # only for logging, could be left out
            'chosen_network_by': "not specified"  # only for logging
        }
        args.evaluation_path = os.path.dirname(args.evaluation_path)
    else:
        # parse the evaluation file from the training to find the best
        # performing network by validation measures and extract the important
        # parameters (model and threshold)
        trainings_settings_and_results = parse_train_and_valid_results(
            args.evaluation_path,
            args.choose_best_epoch_by,
            args.detection_measure
        )

    tasks = trainings_settings_and_results['tasks']

    # get preprocessing
    data_transform, _, _ = \
        get_preprocessing(trainings_settings_and_results['model'])

    # get datasets and dataloaders independently
    dataset_loaders = {}
    datasets = {}
    for dataset_name in dataset_names:
        datasets[dataset_name] = \
            get_combined_dataset(dataset_name,
                                 set_name='test',
                                 transform=data_transform,
                                 basepath=args.dataset_basepath)[0]
        dataset_loaders[dataset_name] = \
            torch.utils.data.DataLoader(datasets[dataset_name],
                                        shuffle=False,
                                        num_workers=2,
                                        batch_size=args.batch_size,
                                        pin_memory=True,
                                        drop_last=False)

    # load network
    # use CUDA if possible
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = get_model_by_string(trainings_settings_and_results['model'],
                                device=device)
    # load weights
    state_dict = torch.load(
        trainings_settings_and_results['weights_filepath'],
        map_location=device
    )
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    softmax = torch.nn.Softmax(dim=1).to(device=device)

    # inference
    results = {}
    for dataset_name in dataset_names:
        data_metainfos = []

        # Iterate over the dataset
        print(f"Running evaluation for dataset: '{dataset_name}'")
        for batch_idx, batch_data in tqdm(
                enumerate(dataset_loaders[dataset_name]),
                total=((len(datasets[dataset_name])+args.batch_size) // args.batch_size)):

            input_images = batch_data['image'].to(device, non_blocking=True)

            # Remember all but the image data
            metainfo_dict = {i: batch_data[i] for i in ['dataset', 'tape_name',
                                                        'image_id',
                                                        'instance_id',
                                                        'person_id',
                                                        'walk_pattern_type',
                                                        'category_name']}
            metainfo_dict['is_person'] = batch_data['is_person'].clone()
            metainfo_dict['labels_orientation'] = \
                batch_data['orientation'].clone()
            metainfo_dict['labels_detection'] = batch_data['is_person'].clone()
            metainfo_dict['labels_pose'] = batch_data['pose'].clone()

            # inference without gradient tracking
            with torch.set_grad_enabled(False):
                output = model(input_images)

                predictions = {
                    'detection': softmax(output[:, 0:2]),
                    'orientation': normalize_orientation_output(output[:, 2:4]),
                    'pose': softmax(output[:, 4:7])
                }

            if 'detection' in tasks:
                metainfo_dict['scores_detection'] = \
                    predictions['detection'].cpu()
            if 'orientation' in tasks:
                metainfo_dict['scores_orientation'] = \
                    predictions['orientation'].cpu()
            if 'pose' in tasks:
                metainfo_dict['scores_pose'] = predictions['pose'].cpu()

            data_metainfos.append(metainfo_dict)

        # concatenate all results
        all_metainfos = {}
        for key in data_metainfos[0].keys():
            all_metainfos[key] = []
            for batch in range(len(data_metainfos)):
                all_metainfos[key].append(data_metainfos[batch][key])
            all_metainfos[key] = np.concatenate(all_metainfos[key])

        results[dataset_name] = {
            'chosen_epoch_for_evaluation': trainings_settings_and_results['epoch'],
            'chosen_epoch_by': trainings_settings_and_results['chosen_network_by'],
            'best_epoch': trainings_settings_and_results['epoch'],
            'evaluated_values': all_metainfos,
        }

        # detection evaluation
        if 'detection' in tasks:
            gt = np.array(all_metainfos['labels_detection'], dtype='int')
            pred = np.array(all_metainfos['scores_detection'])

            thresh = trainings_settings_and_results[f'detection_valid_{args.detection_measure}_thresh']
            results[dataset_name][f'detection_test_{args.detection_measure}_thresh'] = thresh

            # get measures (Receiver Operating Characteristics and Precision Recall)
            statistics = get_statistics_binary(gt, pred,
                                               additional_thresholds=(thresh,))
            roc_results = roc_measures(statistics)
            results[dataset_name]['detection_test_roc_results'] = roc_results
            pr_results = pr_measures(statistics)
            results[dataset_name]['detection_test_pr_results'] = pr_results

            # roc measures for threshold extracted from validation data
            index = np.where(roc_results['threshold'] == thresh)[0][0]
            results[dataset_name]['detection_test_accuracy'] = roc_results['accuracy'][index]
            results[dataset_name]['detection_test_balanced_accuracy'] = 1 - roc_results['balanced_error_rate'][index]

            # pr measures for threshold extracted from validation data
            index = np.where(pr_results['threshold'] == thresh)[0][0]
            results[dataset_name]['detection_test_f1'] = pr_results['f1_score'][index]

        # orientation evaluation
        if 'orientation' in tasks:
            pred = np.array(biternion2deg(all_metainfos['scores_orientation']))
            gt = np.array(all_metainfos['labels_orientation'])
            angle_errors = pred - gt
            # mask out all data, where no orientation is labeled, e.g. the multitask dataset
            angle_errors = angle_errors[np.array(all_metainfos['dataset']) == 'orientation']
            # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
            angle_errors = (angle_errors + 180.0) % 360.0 - 180.0
            if len(angle_errors >= 0):
                angle_mse = np.mean(np.abs(angle_errors))
                results[dataset_name]['angle_mse'] = angle_mse
            else:
                # dataset without orientation ground truth
                results[dataset_name]['angle_mse'] = np.nan

        # pose evaluation
        if 'pose' in tasks:
            gt = np.array(all_metainfos['labels_pose'])
            pred = np.argmax(np.array(all_metainfos['scores_pose']), axis=-1)
            mask = gt != -100

            conf = metrics.confusion_matrix(gt[mask], pred[mask],
                                            labels=[0, 1, 2])
            tp = conf.diagonal()
            total = conf.sum(axis=-1)
            accuracy = tp.sum()/total.sum()
            mask = total > 0
            class_accuracies = np.divide(tp, total, where=mask)
            class_accuracies[np.logical_not(mask)] = -1
            bal_accuracy = class_accuracies[mask].mean()
            results[dataset_name]['pose_test_confusion_matrix'] = conf
            results[dataset_name]['pose_test_balanced_accuracy'] = bal_accuracy

    # calculate combined results for multitask and orientation dataset (values
    # used in the paper)
    if 'detection' in tasks:
        if 'multitask' in dataset_names and 'orientation' in dataset_names:
            # calculate measures for combined dataset
            dataset_name = 'multitask+orientation'
            if 'multitask+orientation' not in dataset_names:
                dataset_names.append(dataset_name)
                results[dataset_name] = {}

            # detection
            thresh = trainings_settings_and_results[f'detection_valid_{args.detection_measure}_thresh']
            gt = np.concatenate((results['multitask']['evaluated_values']['labels_detection'],
                                 results['orientation']['evaluated_values']['labels_detection']))
            gt = np.array(gt, dtype='int')
            pred = np.concatenate((results['multitask']['evaluated_values']['scores_detection'],
                                   results['orientation']['evaluated_values']['scores_detection']))

            # get measures
            statistics = get_statistics_binary(gt, pred,
                                               additional_thresholds=(thresh,))
            roc_results = roc_measures(statistics)
            results[dataset_name]['detection_test_roc_results'] = roc_results
            pr_results = pr_measures(statistics)
            results[dataset_name]['detection_test_pr_results'] = pr_results

            # roc measures for valid threshold
            index = np.where(roc_results['threshold'] == thresh)[0][0]
            results[dataset_name]['detection_test_balanced_accuracy'] = 1 - roc_results['balanced_error_rate'][index]

            # pr measures for valid threshold
            index = np.where(pr_results['threshold'] == thresh)[0][0]
            results[dataset_name]['detection_test_f1'] = pr_results['f1_score'][index]

            # pose
            conf = results['multitask']['pose_test_confusion_matrix'].copy()
            conf += results['orientation']['pose_test_confusion_matrix']

            tp = conf.diagonal()
            total = conf.sum(axis=-1)
            mask = total > 0
            class_accuracies = np.divide(tp, total, where=mask)
            class_accuracies[np.logical_not(mask)] = -1
            bal_accuracy = class_accuracies[mask].mean()
            results[dataset_name]['pose_test_confusion_matrix'] = conf
            results[dataset_name]['pose_test_balanced_accuracy'] = bal_accuracy

    # create evaluation folder and store results
    result_dir = os.path.join(args.evaluation_path, 'evaluation')
    if 'detection' in tasks:
        result_fn = (f'test_results_{args.choose_best_epoch_by}_'
                     f'{args.detection_measure}.pkl')
    else:
        result_fn = f'test_results_{args.choose_best_epoch_by}.pkl'
    result_file_path = os.path.join(result_dir, result_fn)

    create_directory_if_not_exists(result_dir)
    with open(result_file_path, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    if 'detection' in tasks:
        for dataset in dataset_names:
            print(f"Detection Test Results for '{dataset}':"
                  f"\n\tf1:\t{results[dataset]['detection_test_f1']} (max: {results[dataset]['detection_test_pr_results']['best_f1_score']})"
                  f"\n\tbacc:\t{results[dataset]['detection_test_balanced_accuracy']}"
                  )

    if 'orientation' in tasks:
        for dataset in dataset_names:
            if 'angle_mse' in results[dataset]:    # ignore iros2020+orientation
                print(f"Orientation Test Results for '{dataset}':"
                      f"\n\tMAE:\t{results[dataset]['angle_mse']}")

    if 'pose' in tasks:
        for dataset in dataset_names:
            print(f"Pose Test Results for '{dataset}':"
                  f"\n\tbacc:\t{results[dataset]['pose_test_balanced_accuracy']}")
    print(f"Evaluated: '{args.evaluation_path}'")


if __name__ == '__main__':
    main()
