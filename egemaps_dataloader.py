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

        with open(pkpath, 'rb') as f:
            # (samples x dimension)
            datamat = pk.load(f)

            self.targets = Variable(to_cu(on_cuda, 
                    torch.LongTensor(np.round(datamat[:,0])))).view(-1)

            self.inputs = Variable(to_cu(on_cuda, 
                    torch.FloatTensor(datamat[:,1:])))

    def __len__(self):
        return np.shape(dataset)[0]

    def __getitem__(self, idx):
        return (self.inputs, self.targets)
