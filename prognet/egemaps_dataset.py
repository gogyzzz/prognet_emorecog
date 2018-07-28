#xxx egemaps_dataset.py

import pickle as pk
import numpy as np
import torch as tc
from torch.autograd import Variable
from torch.utils.data import Dataset

class egemaps_dataset(Dataset):

    def __init__(self, pkpath, cls_wgt, device):

        with open(pkpath, 'rb') as f:
            # (samples x dimension)
            self.datamat = pk.load(f)
            datamat = self.datamat

        print(pkpath, 'shape:', np.shape(datamat)) 
        print('cls_wgt',cls_wgt)

        self.targets = Variable( 
                tc.LongTensor(np.int_(np.round(datamat[:,0])),
                    )).view(-1).to(device)

        self.inputs = Variable( 
                tc.FloatTensor(datamat[:,1:],
                    )).to(device)
        #print(self.inputs, self.targets)

        self.sample_wgt = [cls_wgt[i] for i in 
                np.int_(np.round(datamat[:,0]))]

        #print('check sef of sample_weight:',set(self.sample_wgt))


    def __len__(self):
        return np.shape(self.datamat)[0]

    def __getitem__(self, idx):
        return (self.inputs[idx], self.targets[idx], self.sample_wgt[idx])

