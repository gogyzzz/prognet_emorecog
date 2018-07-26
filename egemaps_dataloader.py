import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle as pk
import numpy as np
import torch.nn as nn

from myscript import to_cu

# Usage
# dataset = egemaps_dataset(dataset_pk_path, bsz, on_cuda)

class egemaps_dataset(Dataset):

    def __init__(self, pkpath, on_cuda):

        self.on_cuda = on_cuda

        with open(pkpath, 'rb') as f:
            self.dataset = to_cu(on_cuda, torch.FloatTensor(pk.load(f))) # (samples x dimension)

    def __len__(self):
        return np.shape(dataset)[0]

    def __getitem__(self, idx):
        return self.dataset[i,:]
