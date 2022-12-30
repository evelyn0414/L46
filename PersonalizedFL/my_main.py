from alg import algs
from util.evalandprint import evalandprint
import os
import torch
import numpy as np
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


def initialize_args(alg="fedavg", device="cpu"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default=alg,
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed ]')
    # parser.add_argument('--datapercent', type=float,
    #                     default=1e-1, help='data percent to use')
    # parser.add_argument('--dataset', type=str, default='pacs',
    #                     help='[vlcs | pacs | officehome | pamap | covid | medmnist]')
    # parser.add_argument('--root_dir', type=str,
    #                     default='./data/', help='data path')
    parser.add_argument('--save_path', type=str,
                        default='./cks/', help='path to save the checkpoint')
    parser.add_argument('--device', type=str,
                        default='cpu', help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--iters', type=int, default=get_nb_max_rounds(100), # 300
                        help='iterations for communication')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    parser.add_argument('--n_clients', type=int,
                        default=NUM_CLIENTS, help='number of clients')
    parser.add_argument('--wk_iters', type=int, default=100, #changed
                        help='optimization iters in local worker between communication')
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')

    # I have no idea of
    # parser.add_argument('--non_iid_alpha', type=float,
    #                     default=0.1, help='data split for label shift')
    # parser.add_argument('--partition_data', type=str,
    #                     default='non_iid_dirichlet', help='partition data way')
    parser.add_argument('--plan', type=int,
                        default=1, help='choose the feature type')
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='init lam, hyperparmeter for metafed')
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
    args = parser.parse_args()
    return args


def train(strategy="fedavg", device="cpu"):
    args = initialize_args(strategy, device)
    SAVE_PATH = os.path.join('./cks/', strategy)

    algclass = algs.get_algorithm_class(strategy)(args, Baseline(), BaselineLoss(), Optimizer)
    # print(algclass)
    best_changed = False
    train_loaders = train_dataset()
    test_loaders = local_test_datasets() + global_test_dataset()
    val_loaders = local_test_datasets()

    best_acc = [0] * NUM_CLIENTS
    best_tacc = [0] * NUM_CLIENTS
    start_iter, n_rounds = 0, get_nb_max_rounds(100)
    wk_iters = 100

    for a_iter in range(start_iter, n_rounds):
        print(f"============ Train round {a_iter} ============")


        res = evaluate_model_on_tests(algclass.server_model, test_loaders, metric)
        print("before training", res)

        if strategy == 'metafed':
            pass
            # for c_idx in range(NUM_CLIENTS):
            #     algclass.client_train(
            #         c_idx, train_loaders[algclass.csort[c_idx]], a_iter)
            # algclass.update_flag(val_loaders)
        else:
            # local client training
            for wi in range(wk_iters):
                for client_idx in range(NUM_CLIENTS):
                    algclass.client_train(client_idx, train_loaders[client_idx], a_iter)

            # server aggregation
            algclass.server_aggre()

        best_acc, best_tacc, best_changed = evalandprint(
            NUM_CLIENTS, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter, best_changed, metric=metric)

    # res = evaluate_model_on_tests(algclass.server_model, test_loaders, metric)
    # print("final result", res)
    # if args.alg == 'metafed':
    #     print('Personalization stage')
    #     for c_idx in range(NUM_CLIENTS):
    #         algclass.personalization(
    #             c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
    #     best_acc, best_tacc, best_changed = evalandprint(
    #         NUM_CLIENTS, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter, best_changed)

    # s = 'Personalized test acc for each client: '
    # for item in best_tacc:
    #     s += f'{item:.4f},'
    # mean_acc_test = np.mean(np.array(best_tacc))
    # s += f'\nAverage accuracy: {mean_acc_test:.4f}'
    # print(s)

    return 0


if __name__ == '__main__':
    train("fedap")