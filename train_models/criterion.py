import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch


class CDTLoss(nn.Module):

    def __init__(self, Delta_list, gamma=0.5, weight=None, reduction=None, device = None):
        super(CDTLoss, self).__init__()
        self.gamma = gamma
        self.Delta_list = torch.pow(torch.FloatTensor(Delta_list), self.gamma)
        self.Delta_list = self.Delta_list.shape[0] * self.Delta_list / torch.sum(self.Delta_list)
        self.Delta_list = self.Delta_list.to(device)
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        if self.reduction == "sum":
            output = x * self.Delta_list
            return F.cross_entropy(output, target, weight=self.weight, reduction='sum')
        else:
            output = x * self.Delta_list
            return F.cross_entropy(output, target, weight=self.weight)


class LDTLoss(nn.Module):

    def __init__(self, Delta_list, gamma = 0.5, weight=None, reduction = None, device = None):
        super(LDTLoss, self).__init__()
        self.gamma = gamma
        self.Delta_list = torch.pow(torch.FloatTensor(Delta_list), self.gamma)
        self.Delta_list = self.Delta_list.shape[0] * self.Delta_list / torch.sum(self.Delta_list)
        self.Delta_list = self.Delta_list.to(device)
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        if self.reduction == "sum":
            ldt_output = (x.T*self.Delta_list[target]).T
            return F.cross_entropy(ldt_output, target, weight=self.weight, reduction = 'sum')
        else:
            ldt_output = (x.T*self.Delta_list[target]).T
            return F.cross_entropy(ldt_output, target, weight=self.weight)
