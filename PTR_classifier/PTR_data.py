#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
import csv

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

def perodict_data(csvfile_name):
    # csvfile_name index=0, header=0, column1= Cu2 Co1 AL1, column2=0
    lines=[]
    with open (csvfile_name, 'r') as fin:
        read=csv.reader(fin)
        for line in read:
            lines.append(line)
    ndata = len(lines)

    pt =  [[-0.1, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.1, -0.1, -0.1, -0.1, -0.1],
          [-0.1, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.1, -0.1, -0.1, -0.1, -0.1],
          [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
          [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
          [-0.1, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]]


    pt_pos={ 'Li': [0, 0], 'Be': [0, 1],  'B': [0, 12],  'C': [0, 13],  'N': [0, 14],  'O': [0, 15], 'F':[0, 16], 
            'Na': [1, 0], 'Mg': [1, 1], 'Al': [1, 12], 'Si': [1, 13],  'P': [1, 14],  'S': [1, 15], 'Cl':[1, 16],
            'K': [2, 0], 'Ca': [2, 1], 'Ga': [2, 12], 'Ge': [2, 13], 'As': [2, 14], 'Se': [2, 15], 'Br':[2,16],
            'Rb': [3, 0], 'Sr': [3, 1], 'In': [3, 12], 'Sn': [3, 13], 'Sb': [3, 14], 'Te': [3, 15], 'I': [3, 16],
            'Cs': [4, 0], 'Ba': [4, 1], 'Tl': [4, 12], 'Pb': [4, 13], 'Bi': [4, 14],
            'Sc': [2, 2], 'Ti': [2, 3],  'V': [2, 4], 'Cr': [2, 5], 'Mn': [2, 6], 'Fe': [2, 7], 'Co': [2, 8], 'Ni': [2, 9], 'Cu': [2, 10], 'Zn': [2, 11],
            'Y': [3, 2], 'Zr': [3, 3], 'Nb': [3, 4], 'Mo': [3, 5], 'Tc': [3, 6], 'Ru': [3, 7], 'Rh': [3, 8], 'Pd': [3, 9], 'Ag': [3, 10], 'Cd': [3, 11],
            'Hf': [4, 3], 'Ta': [4, 4],  'W': [4, 5], 'Re': [4, 6], 'Os': [4, 7], 'Ir': [4, 8], 'Pt': [4, 9], 'Au': [4, 10], 'Hg': [4, 11]}

    # inicialize x and y
    pt = np.array(pt)
    x = [pt for i in range(ndata)]
    x = np.array(x)
    y = np.zeros(shape=(ndata),dtype=float)
    # get x peridic table
    ii = 0
    for line in lines:
        s = line[0].split(' ')
        for i in range(3):
            x[ii][pt_pos[s[i][:-1]][0]][pt_pos[s[i][:-1]][1]] = 1.4
            if s[i][-1] == '2' :
                x[ii][pt_pos[s[i][:-1]][0]][pt_pos[s[i][:-1]][1]] = 2.8
        y[ii] = float(s[3].rstrip())
        ii+=1
    return x, y

def get_train_val_test_loader(x, y, batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    
    total_size = len(x)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)

    x_dims = np.expand_dims(x, axis = 1)
    y_dims = np.expand_dims(y, axis = 1)
    
    x_train, x_test_val, y_train, y_test_val = train_test_split(x_dims, y_dims, stratify=y_dims, train_size=train_size, random_state=42)
    x_train_tensor = torch.Tensor(x_train)
    y_train_tensor = torch.Tensor(y_train)  
    trainset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(trainset, batch_size=batch_size)

    if return_test:   
        x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, stratify=y_test_val, 
                                                        train_size=valid_size, test_size=test_size, random_state=42)
        x_test_tensor = torch.Tensor(x_test)
        y_test_tensor = torch.Tensor(y_test)
        testset = TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = DataLoader(testset, batch_size=batch_size)    
        x_val_tensor = torch.Tensor(x_val)
        y_val_tensor = torch.Tensor(y_val)
        valset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(valset, batch_size=batch_size)
    else:
        x_val_tensor = torch.Tensor(x_test_val)
        y_val_tensor = torch.Tensor(y_test_val)
        valset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(valset, batch_size=batch_size)


    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader