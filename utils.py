# Dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import pickle as pk
import numpy as np
import torch.nn as nn

def to_cu(is_cuda, tensor):

    if is_cuda is True:
        return tensor.cuda()
    else:
        return tensor

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)



import numpy as np
import pickle
import pandas as pd

hdf_path = 'msp_improv_valid.h5'
stat_path = 'msp_improv_lld/neutral_mean_stddev.lld.pk'

def mk_lld_stat(hdf_path, stat_path):

    df = pd.read_hdf(hdf_path,'table')

    neu_lldpks = df.loc[df.loc[:,'cat']=='N','lldpkpath']

    llds = [pickle.load(open(pkpath,'rb')) for pkpath in neu_lldpks]

    len(llds)

    neu_llds = np.concatenate(llds, axis=0)

    np.shape(neu_llds)

    neu_lld_mean = np.mean(neu_llds, axis=0)

    neu_lld_stddev = np.std(neu_llds, axis=0)

    neu_lld_mean_stddev = [neu_lld_mean, neu_lld_stddev]

    with open(stat_path,'wb') as f:
        pickle.dump(neu_lld_mean_stddev, f)

import pickle as pk
import numpy as np
def normalize_lld(src_path, dst_path, stat_path):
    '''
        src_path: pickled np.ndarray
        dst_path: pickled np.ndarray
        stat_path: pickled (np.ndarray, npndarray)
    '''
    with open(stat_path,'rb') as f:
        mean_std = pk.load(f)
        mean, std = mean_std[0], mean_std[1]

    with open(src_path,'rb') as f:
        normed = pk.load(f) - mean / std

    with open(dst_path, 'wb') as f:
        pk.dump(normed,f)

    print(src_path,'->',dst_path)
        

