import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1, BN=False):
        super(Baseline, self).__init__()
        hidden_dim = 32
        if self.BN:
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.bn = torch.nn.BatchNorm1d(hidden_dim)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.BN:
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.fc2(x)
            return torch.sigmoid(x)
        return torch.sigmoid(self.linear(x))

    def getallfea(self, x):
        if self.BN:
            return [self.fc1(x).clone().detach()]
        else:
            return [self.linear(x).clone().detach()]

    def get_sel_fea(self, x):
        if self.BN:
            return self.fc1(x)
        else:
            return self.linear(x)