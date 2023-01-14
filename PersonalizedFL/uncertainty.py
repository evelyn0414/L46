import sys

import torch.nn as nn
from torch.nn import functional as F
# from torchensemble.utils.logging import set_logger
# from torchensemble import VotingClassifier
import torch
from flamby.utils import evaluate_model_on_tests
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# loading data
# 2 lines of code to change to switch to another dataset
from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR, #learning rate
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    Optimizer,
)
# from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset
from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset

# Instantiation of local train set (and data loader)), baseline loss function, baseline model, default optimizer

lossfunc = BaselineLoss()


def get_local_train_loader(center=0):
    train_dataset = FedDataset(center=center, train=True, pooled=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # print(train_loader.__len__())
    # print(next(iter(train_loader)))
    return train_loader


def get_pooled_test_loader():
    test_dataloader = torch.utils.data.DataLoader(
        FedDataset(train = False, pooled = True),
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = 0,
    )
    return test_dataloader

test_loader = get_pooled_test_loader()
# print(test_loader.__len__())
# print(next(iter(test_loader)))

# logger = set_logger('heart_disease')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


# def test(net, is_MCDO=False):
#     print('Start test')
#     class_correct = list(0. for i in range(10))
#     class_total = list(0. for i in range(10))
#     with torch.no_grad():
#         for data in test_loader:
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#             output = 0
#             if is_MCDO:
#                 for i in range(10):
#                     output += net(inputs)/10.
#                 output = torch.log(output)
#             else:
#                 output = net(inputs)
#             _, predicted = torch.max(output, 1)
#             c = (predicted == labels).squeeze()
#             for i in range(len(labels)):
#                 label = labels[i]
#                 class_correct[label] += c[i].item()
#                 class_total[label] += 1
#

def evaluate_model_on_tests(
        model, test_dataloaders, metric, use_gpu=True, return_pred=False, MCDO=False, T=10, Ensemble=False
):

    results_dict = {}
    y_true_dict = {}
    y_pred_dict = {}
    entropy_dict = {}
    variance_dict = {}
    if Ensemble:
        for m in model:
            if torch.cuda.is_available() and use_gpu:
                m = m.cuda()
            m.eval()
    else:
        if torch.cuda.is_available() and use_gpu:
            model = model.cuda()
        model.eval()
    if MCDO:
        enable_dropout(model)
    with torch.no_grad():
        for i in tqdm(range(len(test_dataloaders))):
            test_dataloader_iterator = iter(test_dataloaders[i])
            y_pred_final = []
            y_true_final = []
            entropy_final = []
            variance_final = []
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                print(X.size())
                print(y.size())
                outputs = np.empty((0, list(y.size())[0], list(y.size())[1]))
                if MCDO:
                    # print("MC Dropout enabled with T =", T)
                    for _ in range(T):
                        # output += model(X) / float(T)
                        out = model(X).detach().cpu()
                        outputs = np.vstack((outputs, out[np.newaxis, :, :])) # shape (forward_passes, n_samples, n_classes)
                        # print(outputs)
                elif Ensemble:
                    for m in model:
                        # output += m(X) / float(len(model))
                        out = m(X).detach().cpu()
                        outputs = np.vstack((outputs, out[np.newaxis, :, :])) # shape (forward_passes, n_samples, n_classes)

                if MCDO or Ensemble:
                    # y_pred = output
                    y_pred = np.mean(outputs, axis=0) # shape (n_samples, n_classes)
                    variance = np.var(outputs, axis=0)

                    variance_final.append(variance)
                    
                else:
                    y_pred = model(X).detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                y_pred_final.append(y_pred)
                y_true_final.append(y)
                
                epsilon = sys.float_info.min
                # Calculating entropy across multiple MCD forward passes
                entropy = -np.sum(y_pred *np.log(y_pred + epsilon), axis=-1) # shape (n_samples,)
                entropy_final.append(entropy)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            if MCDO or Ensemble:
                variance_final = np.concatenate(variance_final)
                variance_dict[f"client_test_{i}"] = variance_final
            entropy_final = np.concatenate(entropy_final)
            results_dict[f"client_test_{i}"] = metric(y_true_final, y_pred_final)
            if return_pred:
                y_true_dict[f"client_test_{i}"] = y_true_final
                y_pred_dict[f"client_test_{i}"] = y_pred_final
                entropy_dict[f"client_test_{i}"] = entropy_final
    if return_pred:
        return results_dict, y_true_dict, y_pred_dict, variance_dict, entropy_dict
    else:
        return results_dict


def plot_box(data, x_ticks, title=None):
    plt.style.use('ggplot')
    # fig, ax = plt.subplots(figsize=(16,9))
    fig, ax = plt.subplots()
    ax.boxplot(data)
    # ax.bar(x_ticks, [np.mean(row) for row in data])
    ax.set_xticklabels(x_ticks)
    ax.set_title(title)
    # ax.set_yscale("log")
    ax.tick_params(axis='x', rotation=10)
    # plt.show()
    plt.savefig("figs/MCD-1000/" + title + ".png")


def evaluate(dict_cindex, y_true, y_pred, variance, entropy, uncertainty=True, id=""):
    """
    only valid for binary classification now, due to the correct/wrong thing
    """
    print(dict_cindex)
    for k in y_true:
        print(k)
        correct_var, wrong_var, correct_ent, wrong_ent = [], [], [], []
        for i in range(len(y_true[k])):
            if (y_pred[k][i] > 0.5) == y_true[k][i]:
                if uncertainty:
                    correct_var.append(variance[k][i][0])
                correct_ent.append(entropy[k][i])
            else:
                if uncertainty:
                    wrong_var.append(variance[k][i][0])
                wrong_ent.append(entropy[k][i])
        if uncertainty:
            print(np.mean(correct_var) )
            print(np.mean(wrong_var))
            plot_box([correct_var, wrong_var], ["correct", "wrong"], "model_{}_{}".format(str(id), k) + "_variance")
        print(np.mean(correct_ent))
        print(np.mean(wrong_ent))
        plot_box([correct_ent, wrong_ent], ["correct", "wrong"], "model_{}_{}".format(str(id), k) + "_Entropy")


def try_baseline(center=0):
    lossfunc = BaselineLoss()
    model = Baseline()
    optimizer = Optimizer(model.parameters(), lr=LR)
    train_loader = get_local_train_loader(center=center)
    for epoch in range(0, NUM_EPOCHS_POOLED):
        for idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X)
            loss = lossfunc(outputs, y)
            loss.backward()
            optimizer.step()
    test_dataloaders = [
        torch.utils.data.DataLoader(
            FedDataset(center=center, train=False, pooled=False),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        ),
        torch.utils.data.DataLoader(
            FedDataset(train = False, pooled = True),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        )
    ]
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(model, test_dataloaders, metric, return_pred=True, T=100)
    evaluate(dict_cindex, y_true, y_pred, variance, entropy, uncertainty=False, id=center)


def try_MC(T=100, center=0):
    # try out MC-dropout
    lossfunc = BaselineLoss()
    model = Baseline(MCDO=True)
    optimizer = Optimizer(model.parameters(), lr=LR)
    train_loader = get_local_train_loader(center=center)
    print("using dropout")
    for epoch in range(0, NUM_EPOCHS_POOLED):
        for idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X)
            loss = lossfunc(outputs, y)
            loss.backward()
            optimizer.step()

    test_dataloaders = [
        torch.utils.data.DataLoader(
            FedDataset(center=center, train=False, pooled=False),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        ),
        torch.utils.data.DataLoader(
            FedDataset(train = False, pooled = True),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        )
    ]
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(model, test_dataloaders, metric, MCDO=True, return_pred=True, T=T)
    evaluate(dict_cindex, y_true, y_pred, variance, entropy, id=center)


def try_ensemble(num_models=10, center=0):
    print("using ensemble")
    models = [Baseline().to(device) for _ in range(num_models)]
    lossfunc = BaselineLoss()

    for model in models:
        train_loader = get_local_train_loader(center=center)
        optimizer = Optimizer(model.parameters(), lr=LR)
        for epoch in range(0, NUM_EPOCHS_POOLED):
            for idx, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(X)
                loss = lossfunc(outputs, y)
                loss.backward()
                optimizer.step()

    test_dataloaders = [
        torch.utils.data.DataLoader(
            FedDataset(center=center, train=False, pooled=False),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        ),
        torch.utils.data.DataLoader(
            FedDataset(train = False, pooled = True),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        )
    ]
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(models, test_dataloaders, metric, Ensemble=True, return_pred=True)
    evaluate(dict_cindex, y_true, y_pred, variance, entropy, id=center)


# def try_ensemble_old():
#     model = VotingClassifier(
#         estimator=Baseline(),
#         n_estimators=1,
#         cuda=False,
#     )

#     criterion = nn.BCELoss()
#     model.set_criterion(criterion)
#     train_loader = get_local_train_loader(center=0)
#     # model.set_criterion(BaselineLoss)

#     model.set_optimizer('Adam',  # parameter optimizer
#                         lr=LR,  # learning rate of the optimizer
#                         weight_decay=5e-4)  # weight decay of the optimizer

#     # Training
#     model.fit(train_loader=train_loader,  # training data
#               epochs=NUM_EPOCHS_POOLED)                 # the number of training epochs

#     # Evaluating
#     accuracy = model.predict(test_loader)
#     print(accuracy)


if __name__ == '__main__':
    # try_ensemble(1, 1)
    # try_ensemble(10, 1)
    # try_MC(False, 1)
    try_MC(1000, 1)
    # try_baseline(0)
    # try_ensemble_old()
