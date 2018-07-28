#!/usr/bin/env python 
import os
import json as js
import argparse as argp

import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader

from prognet.get_class_weight import get_class_weight
from prognet.egemaps_dataset import egemaps_dataset
from prognet.forward_lazy import validate_war_lazy 
from prognet.forward_lazy import validate_uar_lazy
from prognet.prognet import prognet
from prognet.prognet import dnn
from prognet.train import train
from prognet.train import validate_loop_lazy

device=tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

pars = argp.ArgumentParser()
pars.add_argument('--propjs', help='property json')
with open(pars.parse_args().propjs) as f:
    #p = js.load(f.read())
    p = js.load(f)

print(js.dumps(p, indent=4))

pretrain_pk = p['dataset']+'/pretrain.pk'
train_pk    = p['dataset']+'/train.pk'
predev_pk   = p['dataset']+'/predev.pk'
dev_pk      = p['dataset']+'/dev.pk'
eval_pk     = p['dataset']+'/eval.pk'

dnn_pth = p['premodel']
prognet_pth    =  p['model']

lr=p['lr']
preephs=p['pre_ephs']
ephs=p['ephs']
bsz=p['bsz']

with open(p['dataset']+'/idx_prelabel.json') as f:
    dnn_cls = js.load(f)
    
with open(p['dataset']+'/idx_label.json') as f:
    prognet_cls = js.load(f)

nin=p['nin']
nhid=p['nhid']
measure=p['measure']

dnn_nout = len(dnn_cls)
prognet_nout = len(prognet_cls)

dnn_cls_wgt = get_class_weight(pretrain_pk,device)
prognet_cls_wgt = get_class_weight(train_pk,device)

valid_lazy = {'uar': validate_uar_lazy,
                'war': validate_war_lazy }

# loading
pretrainloader = DataLoader(egemaps_dataset(pretrain_pk, dnn_cls_wgt,device), bsz)
predevloader = DataLoader(egemaps_dataset(predev_pk, dnn_cls_wgt,device), bsz)
trainloader = DataLoader(egemaps_dataset(train_pk, prognet_cls_wgt,device), bsz)
devloader = DataLoader(egemaps_dataset(dev_pk, prognet_cls_wgt,device), bsz)
evalloader = DataLoader(egemaps_dataset(eval_pk, prognet_cls_wgt,device), bsz)

# pretraining
dnn_mdl = dnn(nin, nhid, dnn_nout)
dnn_mdl.to(device)

if os.path.exists(dnn_pth):
    print(dnn_pth, 'already exists.')
else:

    optim = tc.optim.Adam(dnn_mdl.parameters())

    precrit = nn.CrossEntropyLoss(weight=dnn_cls_wgt)

    _val_lz = valid_lazy[measure](crit=precrit)
    _val_loop_lz = validate_loop_lazy(name='valid', loader=predevloader)

    pretrained = train( dnn_mdl, pretrainloader, _val_lz, _val_loop_lz, precrit, optim )

    tc.save(pretrained.state_dict(), dnn_pth)

# training
dnn_mdl.load_state_dict(tc.load(dnn_pth))

prog_mdl = prognet(dnn_mdl, nin, nhid, prognet_nout)
prog_mdl.to(device)
crit = nn.CrossEntropyLoss(weight=prognet_cls_wgt)

if os.path.exists(prognet_pth):
    print(prognet_pth, 'already exists')

else:

    optim = tc.optim.Adam(prog_mdl.parameters(), lr=0.00005)

    _val_lz = valid_lazy[measure](crit=crit)
    _val_loop_lz = validate_loop_lazy(name='valid', loader=devloader)

    trained = train(prog_mdl, trainloader, _val_lz, _val_loop_lz, crit, optim)

    tc.save(trained.state_dict(), prognet_pth)

prog_mdl.load_state_dict(tc.load(prognet_pth))
_val_lz = valid_lazy[measure](model=prog_mdl, crit=crit)

# test
validate_loop_lazy('test', _val_lz, evalloader) 

