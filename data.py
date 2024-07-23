import torch
import pandas as pd
import scipy.io
import collections
import os
from os import listdir
from os.path import isfile, join
from random import randint
import numpy as np
from torch import Tensor
from torch.utils.data._utils.collate import default_collate as collate_fn
import math

def mat_name_parser(filename):
    ghi, temperature, _ = filename.replace('_', '.').split('.')
    temperature = float(temperature)/10
    parsed = r'$GHI = {} W/m^2$, $T={:.1f}$'.format(ghi, temperature)
    return parsed

def mat_name_parser2(filename):
    name = filename.replace('R=', '').replace('L=', '').replace('C=', '').replace('.mat', '').replace('_DIBS', '')
    r, l, c = name.split(',')
    r, l, c = float(r), float(l), float(c)
    parsed = 'R={},L={},C={}'.format(r, l, c)
    return parsed

def load_mat(file_path, data_head):
    file = scipy.io.loadmat(file_path)
    data = file[data_head][:]
    df = pd.DataFrame(data)
    return df

class MetaTask:
    def __init__(self, data_train, data_test, idx=None):
        self.data_train = data_train
        self.data_test = data_test
        self.idx = idx

    def __iter__(self):
        return iter((self.data_train, self.data_test))

def zdataset(path, head='data', parser=mat_name_parser,
             transform_after=lambda x: x,
             ):
    df = load_mat(path, head)

    df = np.array(df, dtype=np.single)
    name = os.path.basename(path)
    name = parser(name)
    return transform_after(df), name

class MetaZDataProvider(collections.abc.Iterable):
    def __init__(self, dataset=None, root='./data', length=20000,
                 batch_size=16, batch_size_eval=12,
                 sampler='random',
                 parser=mat_name_parser,
                 items=None,
                 ):
        self.length = length
        self.batch_size = batch_size
        files = [f for f in listdir(root) if isfile(join(root, f))]
        files.sort()
        if items is not None:
          files = files[:items]
        self.datasets = [zdataset(root + '/' + file, parser=parser) for file in files]
        self.categorys = len(self.datasets)
        self.batch_size_eval = batch_size_eval
        self.sampler = sampler

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(self.length):
            if self.sampler == 'random':
                dataset_idx = randint(0, self.categorys - 1)
            else:
                dataset_idx = i % self.categorys
            dataset, name = self.datasets[dataset_idx]
            indices = self.batch_size if isinstance(self.batch_size, list) \
                else np.random.randint(len(dataset), size=self.batch_size)
            indices_eval = self.batch_size_eval if isinstance(self.batch_size_eval, list) \
                else np.random.randint(len(dataset), size=self.batch_size_eval)
            train = (torch.tensor(dataset[indices][:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] ]), torch.tensor(dataset[indices][:, [0]]))
            test = (torch.tensor(dataset[indices_eval][:, [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]), torch.tensor(dataset[indices_eval][:, [0]]))
            yield MetaTask(train, test, idx=name)

