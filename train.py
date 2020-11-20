#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
from datetime import datetime
import os
import pickle
import sys
import warnings

import numpy as np
import sklearn.metrics as metrics
import torch

from src.data.dataset import get_combined_dataset
from src.data.dataset_sampler import DatasetSamplerMultiDataset
from src.data.postprocessing import biternion2deg
from src.data.postprocessing import normalize_orientation_output
from src.data.preprocessing import OrientationAugmentation
from src.data.preprocessing import get_preprocessing
from src.evaluation_utils import get_statistics_binary
from src.evaluation_utils import pr_measures
from src.evaluation_utils import roc_measures
from src.io_utils import create_directory_if_not_exists
from src.logger import CSVLogger
from src.losses import get_new_loss_weights_dwa
from src.losses import VonmisesLossBiternion
from src.lr_decay import LRPolyDecay
from src.models import get_model_by_string
from src.parameters import add_hyperparameters_to_argparser


def _parse_args():
    """Parse command-line arguments"""
    desc = 'Train neural network for multi-task person perception'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    # dataset -----------------------------------------------------------------
    parser.add_argument('-db', '--dataset_basepath',
                        type=str,
                        default='./datasets',
                        help='Path to downloaded dataset')

    parser.add_argument('-ds', '--datasets',
                        type=str,
                        default='multitask+orientation',
                        choices=['multitask+orientation',
                                 'multitask',
                                 'orientation'],
                        help='Datasets to use seperated by +')

    parser.add_argument('-rd', '--result_dir',
                        type=str,
                        default='./results',
                        help='Where to store the results')

    # hyper parameters --------------------------------------------------------
    parser = add_hyperparameters_to_argparser(parser)

    # return parsed args
    return parser.parse_args()


def main():
    args = _parse_args()
    batch_size = args.batch_size
    dataset_names = args.datasets.split('+')
    tasks = args.tasks.split('+')
    tasks.sort()    # sort so we always have the same order

    print('Loading Dataset from ' + args.dataset_basepath)
    print('Using datasets: ' + ' '.join(dataset_names))
    print('Tasks: ' + ' '.join(tasks))

    if 'orientation' in tasks and 'orientation' not in dataset_names:
        warnings.warn("No ground-truth data for orientation in datasets")
    if 'detection' in tasks and 'multitask' not in dataset_names:
        warnings.warn("No non-person data available in datasets")
    if 'pose' in tasks and 'multitask' not in dataset_names:
        warnings.warn("Only standing persons in datasets")

    # create training ID and folder, make sure random id is unique
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    train_dir = os.path.join(args.result_dir,
                             f'{args.training_name}__{training_starttime}')
    if os.path.exists(train_dir):
        raise IOError(f'Output directory: {train_dir} already exists.')

    create_directory_if_not_exists(train_dir)
    model_dir = os.path.join(train_dir, 'models')
    create_directory_if_not_exists(model_dir)
    network_outputs_dir = os.path.join(train_dir, 'network_outputs')
    create_directory_if_not_exists(network_outputs_dir)

    # get preprocessing
    data_transform, _, _ = get_preprocessing(args.model)

    # augmentation
    if args.augmentation == 'flip':
        augmentation = OrientationAugmentation()
    else:
        augmentation = None

    # train data
    dataset_list_train = get_combined_dataset(dataset_names,
                                              set_name='train',
                                              transform=data_transform,
                                              basepath=args.dataset_basepath,
                                              augmentation=augmentation)
    dataset_train = torch.utils.data.ConcatDataset(dataset_list_train)

    if args.dataset_combination == 'concat':
        detection_train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=False, drop_last=True
        )
    elif args.dataset_combination == '50_50':
        # create uniform dataset sampler
        sampler = DatasetSamplerMultiDataset(dataset_list_train,
                                             batch_size=batch_size,
                                             shuffle=True)
        detection_train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True
        )
    else:
        raise ValueError(f"Unknown dataset combination method "
                         f"{args.dataset_combination}")

    # validation data
    dataset_list_valid = get_combined_dataset(dataset_names,
                                              set_name='valid',
                                              transform=data_transform,
                                              basepath=args.dataset_basepath)
    dataset_valid = torch.utils.data.ConcatDataset(dataset_list_valid)

    detection_valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=2*batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=False, drop_last=False)

    # Store the data loaders
    n_batches_train = len(dataset_train) // batch_size
    dataset_loaders = {
        'train': detection_train_loader,
        'valid': detection_valid_loader
    }

    # load network
    # use CUDA if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = get_model_by_string(args.model, device)
    softmax = torch.nn.Softmax(dim=1)

    # optimizer
    params_lr = [{'params': model.parameters(), 'lr': args.learning_rate}]

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_lr, momentum=args.momentum)
        if args.model.startswith('mobilenetv2'):
            warnings.warn("\n\nMobileNetV2 should be trained with Adam\n\n")
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_lr)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not yet "
                                  f"implemented")

    # losses and loss weights
    criterions = {
        'detection': torch.nn.CrossEntropyLoss(reduction='none'),
        'orientation': VonmisesLossBiternion(kappa=args.kappa),
        'pose': torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    }

    weights = {'detection': args.weight_detection,
               'orientation': args.weight_orientation,
               'pose': args.weight_pose}
    weights_loss_history = {'detection': [],
                            'orientation': [],
                            'pose': []}

    # lr decay
    lr_decay = LRPolyDecay(args.learning_rate,
                           power=args.learning_rate_decay,
                           max_iter=args.n_epochs*n_batches_train,
                           lr_min=1e-6)

    # logging
    csvlogger = CSVLogger(os.path.join(train_dir, 'training.csv'))

    # dump the parameters which were given to the script
    with open(os.path.join(train_dir, 'argument_list.txt'), 'w') as f:
        f.write(' '.join([f'--{k} {v}' for k, v in vars(args).items()]) + '\n')

    # train loop
    for epoch in range(args.n_epochs):    # loop over the dataset
        running_loss_train = 0.0

        # create dicts for labels and scores
        labels = {'train': {}, 'valid': {}}
        scores = {'train': {}, 'valid': {}}
        losses = {'train': {}, 'valid': {}}
        metainfos = {'train': {}, 'valid': {}}
        losses_by_task = {'train': {}, 'valid': {}}
        losses_overall = {'train': 0.0, 'valid': 0.0}

        for phase in ['train', 'valid']:
            for task in tasks:
                labels[phase][task] = []
                scores[phase][task] = []
                losses[phase][task] = []
            for info in ['dataset', 'category_name', 'tape_name', 'person_id',
                         'image_id', 'instance_id']:
                metainfos[phase][info] = []

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # set model to train (dropout enabled etc.)
            else:
                model.eval()  # set model to eval (dropout disabled etc.)
                print('Validation...')

            # iterate through dataset
            for batch_idx, batch_data in enumerate(dataset_loaders[phase]):

                input_images = batch_data['image'].to(device, non_blocking=True)

                targets = {}
                if 'detection' in tasks:
                    targets['detection'] = \
                        batch_data['is_person'].to(device, non_blocking=True).long()

                if 'orientation' in tasks:
                    orientations_deg = np.array(batch_data['orientation'])
                    orientations_bit = \
                        np.array([np.cos(np.deg2rad(orientations_deg)),
                                  np.sin(np.deg2rad(orientations_deg))],
                                 dtype=np.float32)
                    orientations_bit = \
                        torch.from_numpy(orientations_bit.transpose((1, 0)))
                    targets['orientation'] = \
                        orientations_bit.contiguous().to(device,
                                                         non_blocking=True)

                if 'pose' in tasks:
                    targets['pose'] = batch_data['pose'].to(device,
                                                            non_blocking=True)

                # reset gradients
                optimizer.zero_grad()

                # track gradients only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input_images)

                    predictions = {
                        'detection': output[:, 0:2],
                        'orientation': normalize_orientation_output(output[:, 2:4]),
                        'pose': output[:, 4:7]
                    }

                    loss = 0
                    for task in tasks:
                        if task == 'orientation':
                            # mask out only the orientation samples
                            mask = [torch.from_numpy(np.array(batch_data['dataset']) == 'orientation')]
                        elif task == 'pose':
                            # mask out patches without pose
                            mask = targets[task] != -100
                        else:
                            # this will return all elements
                            mask = None

                        loss_ = criterions[task](predictions[task],
                                                 targets[task])

                        # loss for backpropagation
                        loss += weights[task] * loss_[mask].mean()

                        # (unmasked) loss for stats and running mean later
                        losses[phase][task].append(loss_.detach())

                    # run backpropagation and parameter update
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        lr_decay.update_optimizer(optimizer)

                    # statistics
                    # store all batch labels, scores and meta information
                    for task in tasks:
                        labels[phase][task].append(targets[task].clone())
                        scores[phase][task].append(predictions[task].detach())
                    for info in ['dataset', 'category_name', 'tape_name',
                                 'person_id', 'image_id', 'instance_id']:
                        metainfos[phase][info].append(batch_data[info])

                    if phase == 'train':
                        running_loss_train += loss.item()
                        if batch_idx % 10 == 9:
                            # print every 10 mini-batches
                            running_loss_train_mean = \
                                running_loss_train / (batch_idx + 1)
                            print(f'[{(epoch+1):d}, {(batch_idx+1): 5d}] '
                                  f'train_loss: {running_loss_train_mean:.6f}')

            # phase (train or valid) completed
            for task in tasks:
                labels[phase][task] = torch.cat(labels[phase][task])
                scores[phase][task] = torch.cat(scores[phase][task])
                losses[phase][task] = torch.cat(losses[phase][task])
            for info in ['dataset', 'category_name', 'tape_name', 'person_id',
                         'image_id', 'instance_id']:
                metainfos[phase][info] = np.concatenate(metainfos[phase][info])

            # calculate additional losses
            for task in tasks:  # calculate losses by task
                if task == 'orientation':
                    mask = metainfos[phase]['category_name'] == 'person-standing-deeporientation'
                    losses_by_task[phase][task] = losses[phase][task][mask].mean().item()
                elif task == 'pose':
                    # mask out person-Without-Pose
                    mask = labels[phase][task] != -100
                    losses_by_task[phase][task] = losses[phase][task][mask].mean().item()
                else:
                    losses_by_task[phase][task] = losses[phase][task].mean().item()

                losses_overall[phase] += weights[task] * losses_by_task[phase][task]

                if phase == 'train':
                    weights_loss_history[task].append(losses_by_task[phase][task])

            # move everything to cpu and convert to python structures
            for task in tasks:
                labels[phase][task] = labels[phase][task].cpu().numpy().tolist()
                scores[phase][task] = scores[phase][task].cpu().numpy().tolist()
                losses[phase][task] = losses[phase][task].cpu().numpy().tolist()

        # end of epoch reached
        # save weights
        torch.save(model.state_dict(),
                   os.path.join(model_dir, f'epoch_{epoch}.pt'))

        # dump all results
        with open(os.path.join(network_outputs_dir,
                               f'epoch_{epoch}.pkl'), 'wb') as f:
            pickle.dump({'labels': labels,
                         'scores': scores,
                         'losses': losses,
                         'metainfos': metainfos},
                        f, pickle.HIGHEST_PROTOCOL)

        # create logs for csvlogger
        logs = {'train_loss': losses_overall['train'],
                'valid_loss': losses_overall['valid'],
                'weight_detection': weights['detection'],
                'weight_orientation': weights['orientation'],
                'weight_pose': weights['pose']}

        for i, lr in enumerate(lr_decay.get_current_lr()):
            logs[f'lr_{i}'] = lr

        for phase in ['train', 'valid']:
            for task, l in losses_by_task[phase].items():
                logs[f'{phase}_{task}_loss'] = l

        # calculate statistics for validation data
        # detection
        if 'detection' in tasks:
            gt = np.array(labels['valid']['detection'])
            pred = softmax(torch.tensor(scores['valid']['detection'])).numpy()
            statistics_classification = get_statistics_binary(gt, pred)

            roc_classification = roc_measures(statistics_classification)
            ber = roc_classification['best_balanced_error_rate']
            logs['detection_valid_balanced_accuracy'] = 1 - ber[0]
            logs['detection_valid_balanced_accuracy_thresh'] = ber[1]

            pr_classification = pr_measures(statistics_classification)
            f1 = pr_classification['best_f1_score']
            logs['detection_valid_f1'] = f1[0]
            logs['detection_valid_f1_thresh'] = f1[1]

        # orientation
        if 'orientation' in tasks:
            mask = metainfos[phase]['dataset'] == 'orientation'
            gt = np.array(labels['valid']['orientation'])[mask]
            pred = np.array(scores['valid']['orientation'])[mask]

            angle_errors = biternion2deg(pred) - biternion2deg(gt)
            # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
            angle_errors = (angle_errors + 180.0) % 360.0 - 180.0
            angle_mse = np.mean(np.abs(angle_errors))
            logs['orientation_valid_mae'] = angle_mse

        # pose
        if 'pose' in tasks:
            gt = np.array(labels['valid']['pose'])
            pred = np.argmax(np.array(scores['valid']['pose']), axis=-1)
            mask = gt != -100

            conf = metrics.confusion_matrix(gt[mask], pred[mask],
                                            labels=[0, 1, 2])
            class_accuracies = conf.diagonal() / conf.sum(axis=-1)
            bal_accuracy = class_accuracies.mean()
            logs['pose_valid_balanced_accuracy'] = bal_accuracy

        # append epoch for csv logging
        logs['epoch'] = epoch
        csvlogger.write_logs(logs)
        if losses_overall['train'] <= 2*1e-6:
            with open(os.path.join(train_dir, 'early_stopping.csv'), 'w') as f:
                f.write('epoch,running_loss_train\n')
                f.write(f"{epoch},{losses_overall['train']:.10f}")
            break

        # update loss weights
        if args.weight_mode == 'dwa':
            weights = get_new_loss_weights_dwa(
                cur_weights=weights,
                loss_history=weights_loss_history,
                tasks=tasks,
                momentum=args.weight_mode_dwa_momentum,
                t=args.weight_mode_dwa_temperature
            )
    # all done
    print('Finished Training')


if __name__ == '__main__':
    main()
