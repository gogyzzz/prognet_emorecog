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
