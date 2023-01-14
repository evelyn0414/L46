from alg import algs
import os
import torch
import numpy as np
from tqdm import tqdm
from flamby.datasets.fed_isic2019 import (
# from flamby.datasets.fed_ixi import (
# from flamby.datasets.fed_camelyon16 import (
# from flamby.datasets.fed_heart_disease import (
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
dataset_name = "fed_ixi"
# dataset_name = "fed_isic2019"
# dataset_name = "fed_heart_disease"
ROUND_PER_SAVE = 1
from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset
# from flamby.datasets.fed_isic2019 import FedIsic2019 as FedDataset
# from flamby.datasets.fed_ixi import FedIXITiny as FedDataset

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
            num_workers = 0,
            drop_last=True,
        )
        for i in range(NUM_CLIENTS)
    ]


def initialize_args(alg="fedavg", device="cpu"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default=alg,
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed ]')
    parser.add_argument('--dataset', type=str, default="fed-heart-disease",
                        help='Dataset to choose: [fed-heart-disease | fed-isic2019 | fed_camelyon16]')
    parser.add_argument('--save_path', type=str,
                        default='./cks/', help='path to save the checkpoint')
    parser.add_argument('--device', type=str,
                        default=device, help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--iters', type=int, default=get_nb_max_rounds(NUM_EPOCHS_POOLED), # 300
                        help='iterations for communication')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    parser.add_argument('--n_clients', type=int,
                        default=NUM_CLIENTS, help='number of clients')
    parser.add_argument('--wk_iters', type=int, default=NUM_EPOCHS_POOLED, #changed from 1
                        help='optimization iters in local worker between communication')
    parser.add_argument('--nosharebn', action='store_true', default=True,
                        help='not share bn')

    parser.add_argument('--plan', type=int,
                        default=1, help='choose the feature type')
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-2,
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
    SAVE_PATH = os.path.join('./cks/', dataset_name + "_" + strategy)
    SAVE_LOG = os.path.join('./cks/', "log_" + dataset_name + "_" + strategy)

    algclass = algs.get_algorithm_class(strategy)(args, Baseline(BN=True), BaselineLoss(), Optimizer)
    train_loaders = train_dataset()
    test_loaders = local_test_datasets() + global_test_dataset()
    val_loaders = local_test_datasets()

    start_iter, n_rounds = 0, get_nb_max_rounds(NUM_EPOCHS_POOLED)
    wk_iters = NUM_EPOCHS_POOLED
    logs, client_logs = [],  [[] for _ in range(NUM_CLIENTS)]

    print("strategy:", strategy)
    if os.path.exists(SAVE_PATH):
        logs, client_logs = load_model(SAVE_PATH, algclass)
        start_iter = len(logs)
        print(f"============ Loaded model from train round {start_iter} ============")
        for i, l in enumerate(logs):
            print("round", i)
            print(l)
        print(client_logs)

    if strategy == 'fedap':
        algclass.set_client_weight(train_loaders)
    elif args.alg == 'metafed':
        algclass.init_model_flag(train_loaders, val_loaders)
        args.iters = args.iters-1
        print('Common knowledge accumulation stage')

    res = evaluate_model_on_tests(algclass.server_model, test_loaders, metric)
    print("before training", res)

    for a_iter in range(start_iter, n_rounds):
        print(f"============ Train round {a_iter} ============")

        if strategy == 'metafed':
            for c_idx in range(NUM_CLIENTS):
                algclass.client_train(
                    c_idx, train_loaders[algclass.csort[c_idx]], a_iter)
            algclass.update_flag(val_loaders)
        else:
            # local client training
            for wi in tqdm(range(wk_iters)):
                for client_idx in range(NUM_CLIENTS):
                    algclass.client_train(client_idx, train_loaders[client_idx], a_iter)

            # server aggregation
            algclass.server_aggre()

        res = evaluate_model_on_tests(algclass.server_model, test_loaders, metric)
        logs.append(res)
        print("performance of server model", res)
        for i, tmodel in enumerate(algclass.client_model):
            client_res = evaluate_model_on_tests(tmodel, [test_loaders[i]], metric)
            performance = client_res["client_test_0"]
            print(f"result for client model {i}:", performance)
            client_logs[i].append(performance)

        if a_iter % ROUND_PER_SAVE == 0:
            print(f' Saving the local and server checkpoint to {SAVE_PATH}')
            tosave = {'current_epoch': a_iter, 'current_metric': res[f"client_test_{NUM_CLIENTS}"], 'logs': np.array(logs), "client_logs": np.array(client_logs)}
            for i,tmodel in enumerate(algclass.client_model):
                tosave['client_model_'+str(i)]=tmodel.state_dict()
            tosave['server_model']=algclass.server_model.state_dict()
            torch.save(tosave, SAVE_PATH)
            tosave_log = {'server_logs': logs, 'client_logs': client_logs}
            torch.save(tosave_log, SAVE_LOG)

    if args.alg == 'metafed':
        print('Personalization stage')
        for c_idx in range(NUM_CLIENTS):
            algclass.personalization(
                c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
            res = evaluate_model_on_tests(algclass.client_model[c_idx], [test_loaders[c_idx]], metric)
            print(f"final result for client {c_idx}:", res["client_test_0"])

    print(logs)
    print(client_logs)


def load_model(SAVE_PATH, algclass):
    print(f"============ Loading model ============")
    loaded = torch.load(SAVE_PATH, map_location=torch.device('cpu'))
    algclass.server_model.load_state_dict(loaded["server_model"])
    test_loaders = global_test_dataset()
    res = evaluate_model_on_tests(algclass.server_model, test_loaders, metric)
    print("server performance:", res["client_test_0"])
    test_loaders = local_test_datasets()
    for i, tmodel in enumerate(algclass.client_model):
        tmodel.load_state_dict(loaded['client_model_'+str(i)])
        res = evaluate_model_on_tests(tmodel, [test_loaders[i]], metric)
        print(f"performance for client {i}:", res["client_test_0"])
    return loaded["logs"].tolist(), loaded["client_logs"].tolist()


def load_log(SAVE_LOG):
    loaded = torch.load(SAVE_LOG, map_location=torch.device('cpu'))
    logs = loaded["server_logs"]
    client_logs = loaded["client_logs"]
    return logs, client_logs


def calculate_stable_performance(logs, client_logs):
    window_size = 3
    client_num = len(client_logs)
    client_res = []
    server_res = np.mean([log[f"client_test_{client_num}"] for log in logs[-window_size:]])
    for client in range(client_num):
        client_res.append(np.mean(client_logs[client][-window_size:]))
    return [server_res] + client_res
    # return server_res, client_res


def get_res_from_log(strategy):
    # print(dataset_name, strategy)
    SAVE_LOG = os.path.join('./cks/', "log_" + dataset_name + "_" + strategy)
    logs, client_logs = load_log(SAVE_LOG)
    res = calculate_stable_performance(logs, client_logs)
    for c in res:
        print(c, end="\t")
    print(np.mean(res[1:]), end="\t")
    print(np.std(res[1:]), end="\t")
    print("")


def MCD_evaluation(device="cpu"):
    from uncertainty import evaluate_model_on_tests, evaluate
    strategy = "fedavg"
    args = initialize_args(strategy, device)
    SAVE_PATH = os.path.join('./cks/', dataset_name + "_" + strategy)
    SAVE_LOG = os.path.join('./cks/', "log_" + dataset_name + "_" + strategy)

    algclass = algs.get_algorithm_class(strategy)(args, Baseline(), BaselineLoss(), Optimizer)
    test_loaders = local_test_datasets() + global_test_dataset()

    logs, client_logs = load_model(SAVE_PATH, algclass)

    print(f"============ MC-Dropout Test ============")
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(algclass.server_model, test_loaders, metric, MCDO=True, T=1000, return_pred=True)
    evaluate(dict_cindex, y_true, y_pred, variance, entropy, id="server")

    for i, tmodel in enumerate(algclass.client_model):
        dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(tmodel, [test_loaders[i]], metric, MCDO=True, T=1000, return_pred=True)
        evaluate(dict_cindex, y_true, y_pred, variance, entropy, id=i)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--alg', type=str, default="fedavg",
    #                     help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed ]')
    # args = parser.parse_args()
    # train(args.alg, "cuda" if torch.cuda.is_available() else "cpu")
    # train("fedap", "cuda" if torch.cuda.is_available() else "cpu")
    # for alg in "fedavg | fedbn | fedprox | fedap | metafed".split(" | "):
    #     train(alg, "cuda" if torch.cuda.is_available() else "cpu")

    # get_res_from_log("fedavg"),
    # get_res_from_log("fedprox"),
    # get_res_from_log("fedbn"),
    # get_res_from_log("fedap")
    get_res_from_log("fedap")
    # get_res_from_log("metafed")

    # MCD_evaluation()
    # model = Baseline()
    # from alg.fedap import get_form
    # get_form(model)
    # print(model)
    # # load_log("./cks/log_fed_heart_disease_fedavg")


    # print(get_nb_max_rounds(NUM_EPOCHS_POOLED), NUM_EPOCHS_POOLED)