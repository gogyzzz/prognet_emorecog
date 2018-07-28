
from sklearn.utils.class_weight import compute_class_weight
import torch as tc
import pickle as pk
import numpy as np

def get_class_weight(dataset_pk, device):
    with open(dataset_pk,'rb') as f:
        datamat = pk.load(f)

    y = np.int_(np.round(datamat[:,0]))
    cls = list(set(y))

    cls_wgt = compute_class_weight('balanced', cls, y)
    return tc.FloatTensor(cls_wgt).to(device)
