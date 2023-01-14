import torch
from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    get_nb_max_rounds,
    Optimizer
)

import argparse
from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset
from flamby.utils import evaluate_model_on_tests


def global_test_dataset():
    return [
        torch.utils.data.DataLoader(
            FedDataset(train = False, pooled = True),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        )
    ]


def local_test_datasets():
    return [
        torch.utils.data.DataLoader(
            FedDataset(center=i, train=False, pooled=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        for i in range(NUM_CLIENTS)
    ]


def train_dataset():
    return [
        torch.utils.data.DataLoader(
            FedDataset(center = i, train = True, pooled = False),
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 0
        )
        for i in range(NUM_CLIENTS)
    ]


def pretrain_dataset(conf):
    partition_sizes = [
        0.3, 0.7
    ]
    from datautil.datasplit import DataPartitioner
    data_partitioner = DataPartitioner(
        conf,
        global_test_dataset(),
        partition_sizes,
        partition_type="evenly",
        # consistent_indices=False,
    )
    return data_partitioner.use(0)