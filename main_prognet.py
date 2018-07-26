import sys
from torch.utils.data import DataLoader

from egemaps_dataset import egemaps_dataset

# arg parsing

expcase = sys.argv[1]
on_cuda = sys.argv[2]

#read %s/param.json
import json
param = {}
with open(expcase +'/param.json') as f:
    param = json.load(f)

#make dataset
pretrainset = DataLoader(
        egemaps_dataset(param['dataset']+'/pretrain.pk', on_cuda), 
        param['bsz'])

trainset = DataLoader(
        egemaps_dataset(param['dataset']+'/train.pk', on_cuda), 
        param['bsz'])

predevset = DataLoader(
        egemaps_dataset(param['dataset']+'/predev.pk', on_cuda), 
        param['bsz'])

devset = DataLoader(
        egemaps_dataset(param['dataset']+'/dev.pk', on_cuda), 
        param['bsz'])

evalset = DataLoader(
        egemaps_dataset(param['dataset']+'/eval.pk', on_cuda), 
        param['bsz'])


dnn_mdl = dnn(param['premodel'], on_cuda)
optim = torch.optim.Adam(dnn_mdl.parameters())

prognet_mdl = prognet(param['model'], on_cuda)
optim = torch.optim.Adam(dnn_mdl.parameters(), lr=0.00005)

# measure(criterion needs weight for each class)
score_func = measure(param['measure'], on_cuda) # war or uar
train(dnn_mdl, score_func, param['lr'], param['pre_ephs'], param['log'])
train(prognet_mdl, score_func, param['lr'], param['ephs'], param['log'])
test(prognet_mdl, score_func, param['log'])
