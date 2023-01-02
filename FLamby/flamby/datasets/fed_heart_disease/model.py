import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(Baseline, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def getallfea(self, x):
        return self.linear(x).clone().detach()

    def get_sel_fea(self, x):
        return self.linear(x).clone().detach()