#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, classification=False):
        super(Net, self).__init__()
        self.classification = classification
        self.conv1 = nn.Conv2d(1, 96, 3, 1, 1)
        self.conv2 = nn.Conv2d(96, 96, 5, 1, 1)
        self.conv3 = nn.Conv2d(96, 96, 3)
        self.fc1 = nn.Linear(96 * 13 * 1, 192)
        if self.classification:
            self.fc2 = nn.Linear(192, 2)

        else:
            self.fc2 = nn.Linear(192, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        
    def forward(self, x):
        pre_out=[]
        x_conv1 = F.relu(self.conv1(x))
        pre_out.append(x_conv1)
        x_conv2 = F.relu(self.conv2(x_conv1))
        pre_out.append(x_conv2)
        x_conv3 = F.relu(self.conv3(x_conv2))
        pre_out.append(x_conv3)
        x_dim1 = x_conv3.view(-1, 96 * 13 * 1)
        pre_out.append(x_dim1)
        x_fc1 = F.relu(self.fc1(x_dim1))
        pre_out.append(x_fc1)
        if self.classification:
            x_fc1 = self.dropout(x_fc1)
        pre_out.append(x_fc1)
        x_out = self.fc2(x_fc1)
        pre_out.append(x_out)
        if self.classification:
            x_out = self.logsoftmax(x_out)
        return x_out, pre_out

