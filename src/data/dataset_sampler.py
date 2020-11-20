# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import numpy as np
import torch


class DatasetSamplerMultiDataset(torch.utils.data.sampler.Sampler):
    """
    Dataset Sampler which returns the indices for a concatenated dataset
    """
    def __init__(self, dataset_list, batch_size, shuffle, partition=None):
        super(DatasetSamplerMultiDataset)

        if partition is None:
            # use uniform partitioning -> e.g. batch_size = 17, 3 datasets
            # 17 / 3 = 5.66 -> 5
            # 17 - 5 = 12 ... 12 / 2 = 6 -> 6
            # 12 - 6 = 6 ... 6 / 1 = 6 -> 6
            # partition = [5, 6, 6]
            partition = []
            batch_size_to_distribute = batch_size
            for a in reversed(range(len(dataset_list))):
                elements_from_this_dataset = \
                    int(np.floor(batch_size_to_distribute/(a+1)))
                partition.append(elements_from_this_dataset)
                batch_size_to_distribute -= elements_from_this_dataset
        else:
            # partitioning must have exactly the amount of items as we have
            # datasets
            assert len(partition) == len(dataset_list)

            # partitioning must in sum be equal to the batch size
            assert batch_size == np.sum(partition)

        self.dataset_lengths = []
        for dataset in dataset_list:
            self.dataset_lengths.append(len(dataset))
        self.batch_size = batch_size
        self.partition = partition
        self.shuffle = shuffle
        self.nr_datasets = len(self.dataset_lengths)

        # how many batches are possible
        # we do not want to have images twice or more in the batch
        # thus, the batch count depends on the smallest dataset/partition ratio
        min_possible_batches = np.Inf
        for i in range(self.nr_datasets):
            min_possible_batches = np.min(
                [min_possible_batches, self.dataset_lengths[i] / partition[i]]
            )
        self.min_possible_batches = int(np.floor(min_possible_batches))

    def __iter__(self):
        dataset_borders = [0]
        dataset_indices = []

        # create dataset indices by shuffling / using the linear range and
        # shifting it to the right position in the concatenated dataset
        for dataset_length in self.dataset_lengths:
            if self.shuffle:
                dataset_indices.append(list(torch.randperm(dataset_length).numpy() + dataset_borders[-1]))
            else:
                dataset_indices.append(list(np.array(range(dataset_length)) + dataset_borders[-1]))
            dataset_borders.append(dataset_borders[-1] + dataset_length)

        index_list = []
        for batch in range(self.min_possible_batches):
            # append data for all batches
            for dataset in range(self.nr_datasets):
                # for every dataset
                for i in range(self.partition[dataset]):
                    # partition-Elements each
                    # start from the beginning of the lists
                    index_list.append(dataset_indices[dataset].pop(0))

        self.index_list = index_list

        return iter(self.index_list)

    def __len__(self):
        return self.batch_size * self.min_possible_batches
