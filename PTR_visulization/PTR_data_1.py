#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
#from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler


# In[ ]:


class PeriodicTable(Dataset):
    # csvfile_name index=0, header=0, column1= Cu2 Co1 AL1, column2=0
    def __init__(self, csvfile_name):
        self.csvfile_name=csvfile_name
        assert os.path.exists(csvfile_name), 'csvfile_name.csv does not exist!'
        with open (csvfile_name, 'r') as fin:
            reader=csv.reader(fin)
            self.lines = [row for row in reader]
        self.ndata = len(self.lines)


        self.pt_pos={ 'Li': [0, 0], 'Be': [0, 1],  'B': [0, 12],  'C': [0, 13],  'N': [0, 14],  'O': [0, 15], 'F':[0, 16], 
                'Na': [1, 0], 'Mg': [1, 1], 'Al': [1, 12], 'Si': [1, 13],  'P': [1, 14],  'S': [1, 15], 'Cl':[1, 16],
                'K': [2, 0], 'Ca': [2, 1], 'Ga': [2, 12], 'Ge': [2, 13], 'As': [2, 14], 'Se': [2, 15], 'Br':[2,16],
                'Rb': [3, 0], 'Sr': [3, 1], 'In': [3, 12], 'Sn': [3, 13], 'Sb': [3, 14], 'Te': [3, 15], 'I': [3, 16],
                'Cs': [4, 0], 'Ba': [4, 1], 'Tl': [4, 12], 'Pb': [4, 13], 'Bi': [4, 14],
                'Sc': [2, 2], 'Ti': [2, 3],  'V': [2, 4], 'Cr': [2, 5], 'Mn': [2, 6], 'Fe': [2, 7], 'Co': [2, 8], 'Ni': [2, 9], 'Cu': [2, 10], 'Zn': [2, 11],
                'Y': [3, 2], 'Zr': [3, 3], 'Nb': [3, 4], 'Mo': [3, 5], 'Tc': [3, 6], 'Ru': [3, 7], 'Rh': [3, 8], 'Pd': [3, 9], 'Ag': [3, 10], 'Cd': [3, 11],
                'Hf': [4, 3], 'Ta': [4, 4],  'W': [4, 5], 'Re': [4, 6], 'Os': [4, 7], 'Ir': [4, 8], 'Pt': [4, 9], 'Au': [4, 10], 'Hg': [4, 11]}

    # inicialize x and y
        #self.pt = np.array(pt)
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        pt =  [[-0.1, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.1, -0.1, -0.1, -0.1, -0.1],
              [-0.1, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.1, -0.1, -0.1, -0.1, -0.1],
              [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
              [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
              [-0.1, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]]
        x=np.array(pt)
        #print(x)
        line=self.lines[idx]
        s = line[0].split(' ')
        crystal=line[0][:-1].replace(' ', '')
        for i in range(3):
            x[self.pt_pos[s[i][:-1]][0]][self.pt_pos[s[i][:-1]][1]] = 1.4
            if s[i][-1] == '2' :
                x[self.pt_pos[s[i][:-1]][0]][self.pt_pos[s[i][:-1]][1]] = 2.8
        y = float(s[-1].rstrip())
        return x, y, crystal


# In[ ]:


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False,label_1_weight=1,  **kwargs):
    
    total_size=len(dataset)
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
    
    if label_1_weight==1:
        train_sampler = SubsetRandomSampler(indices[:train_size])
        val_sampler = SubsetRandomSampler(indices[-(valid_size + test_size):-test_size])
        if return_test:
            test_sampler = SubsetRandomSampler(indices[-test_size:])
    else:
        weights = [label_1_weight if label == 1 else 1 for data, label, crystal in dataset]
        train_sampler=WeightedRandomSampler(weights,num_samples=train_size, replacement=True)
        val_sampler=WeightedRandomSampler(weights,num_samples=valid_size, replacement=True)
        if return_test:
            test_sampler=WeightedRandomSampler(weights,num_samples=test_size, replacement=True)
    '''
    官方对DataLoader的说明是：“数据加载由数据集和采样器组成，基于python的单、多进程的iterators来处理数据。”
    dataset(Dataset): 传入的数据集
    batch_size(int, optional): 每个batch有多少个样本
    shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
    sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引），需要注意的是，一旦指定了这个参数，
                                      那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）
    num_workers (int, optional): 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
    collate_fn (callable, optional): 将一个list的sample组成一个mini-batch的函数
    pin_memory (bool, optional)： 如果设置为True，那么data loader将会在返回它们之前，
                                将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
    '''
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


# In[ ]:


def collate_pool(dataset_list):
    batch_x,batch_y, batch_crystal=[],[],[]
    for i, (x, y, crystal)in enumerate(dataset_list):
        tensor_x=torch.Tensor(x)
        x_dim=np.expand_dims(tensor_x, axis = 0)
        x_dim=np.expand_dims(x_dim, axis = 0)
        batch_x.append(torch.Tensor(x_dim))
        batch_y.append(torch.Tensor([y]))
        batch_crystal.append(crystal)

    return torch.cat(batch_x, dim=0),            torch.stack(batch_y, dim=0),            batch_crystal

