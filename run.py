#!/usr/bin/env python

##### import package.py

import os
import sys
import pickle as pk
import numpy as np
import json as js
from functools import partial
from toolz import curry

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

##### property.py

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrain_pk = 'iemocap/tmp2/pretrain.pk'
train_pk    = 'iemocap/tmp2/train.pk'
predev_pk   = 'iemocap/tmp2/predev.pk'
dev_pk      = 'iemocap/tmp2/dev.pk'
eval_pk     = 'iemocap/tmp2/eval.pk'

dnn_pth = 'iemocap/tmp2/exp/premodel.pth'
prognet_pth    = 'iemocap/tmp2/exp/model.pth'


lr=0.00005
preephs=150
ephs=300
bsz=512

dnn_cls={0:'Male',1:'Female'}
prognet_cls={0:'Happiness', 1:'Sadness', 2:'Neutral', 3:'Anger'}

nin=88
nhid=256
dnn_nout=len(dnn_cls)
print('dnn_nout',dnn_nout)
prognet_nout=len(prognet_cls)
print('prognet_nout',prognet_nout)

measure='war'

##### egemaps_dataset.py

class egemaps_dataset(Dataset):

    def __init__(self, pkpath, cls_wgt):

        with open(pkpath, 'rb') as f:
            # (samples x dimension)
            self.datamat = pk.load(f)
            datamat = self.datamat

        print(pkpath, 'shape:', np.shape(datamat)) 
        print('cls_wgt',cls_wgt)

        self.targets = Variable( 
                torch.LongTensor(np.int_(np.round(datamat[:,0])),
                    )).view(-1).to(device)

        self.inputs = Variable( 
                torch.FloatTensor(datamat[:,1:],
                    )).to(device)
        #print(self.inputs, self.targets)

        self.sample_wgt = [cls_wgt[i] for i in 
                np.int_(np.round(datamat[:,0]))]

        #print('check sef of sample_weight:',set(self.sample_wgt))


    def __len__(self):
        return np.shape(self.datamat)[0]

    def __getitem__(self, idx):
        return (self.inputs[idx], self.targets[idx], self.sample_wgt[idx])


@curry
def validate_war_lazy(batch, model, crit):
    inputs = batch[0]
    targets = batch[1]
    sample_wgts = batch[2]

    outputs = model(inputs)
    loss = crit(outputs, targets)
    score = recall_score(
        outputs.max(dim=1)[1].data.cpu().numpy(),
        targets.data.cpu().numpy(),
        average='weighted',
        sample_weight=sample_wgts)

    return loss, score

@curry
def validate_uar_lazy(batch, model, crit):
    inputs = batch[0]
    targets = batch[1]

    outputs = model(inputs)
    loss = crit(outputs, targets)

    score = recall_score(
        outputs.max(dim=1)[1].data.cpu().numpy(),
        targets.data.cpu().numpy(),
        average='macro')

    return loss, score

##### prognet.py

def init_linear(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)

class dnn(nn.Module):
    def __init__(self):
        super(dnn, self).__init__()

        #self.bn0 = nn.BatchNorm1d(ninput)

        self.fc1 = nn.Linear(nin, nhid)
        #self.bn1 = nn.BatchNorm1d(nhid)

        self.fc2 = nn.Linear(nhid, nhid)
        #self.bn2 = nn.BatchNorm1d(nhid)

        self.fc3 = nn.Linear(nhid, nhid)
        #self.bn3 = nn.BatchNorm1d(nhid)

        self.fc4 = nn.Linear(nhid, nhid)
        #self.bn4 = nn.BatchNorm1d(nhid)

        self.fc_out = nn.Linear(nhid, dnn_nout)

        self.apply(init_linear)

    def forward(self, x):

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(F.dropout(self.fc2(x)))
        x = F.sigmoid(F.dropout(self.fc3(x)))
        x = F.sigmoid(F.dropout(self.fc4(x)))
        x = F.softmax(self.fc_out(x))

        return x

class prognet(nn.Module):

    def __init__(self, pretrained): 

        super(prognet, self).__init__()

        self.pretrained = pretrained

        self.fc1_w = nn.Linear(nin, nhid)
        #self.bn1 = nn.BatchNorm1d(nhid) # unused

        self.fc2_w = nn.Linear(nhid, nhid)
        self.fc2_u = nn.Linear(nhid, nhid)
        #self.bn2 = nn.BatchNorm1d(nhid)

        self.fc3_w = nn.Linear(nhid, nhid)
        self.fc3_u = nn.Linear(nhid, nhid)
        #self.bn3 = nn.BatchNorm1d(nhid)

        self.fc4_w = nn.Linear(nhid, nhid)
        self.fc4_u = nn.Linear(nhid, nhid)
        #self.bn4 = nn.BatchNorm1d(nhid)

        self.fc_out_w = nn.Linear(nhid, prognet_nout)
        self.fc_out_u = nn.Linear(nhid, prognet_nout) 

        self.apply(init_linear)

    def forward(self, x):
    
        #print(self.pretrained)
        fzly1 = Variable(F.sigmoid(self.pretrained.fc1(x)), 
                requires_grad=False)
        fzly2 = Variable(F.sigmoid(self.pretrained.fc2(fzly1)), 
                requires_grad=False)
        fzly3 = Variable(F.sigmoid(self.pretrained.fc3(fzly2)), 
                requires_grad=False)
        fzly4 = Variable(F.sigmoid(self.pretrained.fc4(fzly3)), 
                requires_grad=False)

        ly1 = F.sigmoid(self.fc1_w(x))

        # progressive neural network eq. (2)

        ly2 = F.sigmoid(
                F.dropout(self.fc2_w(ly1))
                + F.dropout(self.fc2_u(fzly1)))

        ly3 = F.sigmoid(
                F.dropout(self.fc3_w(ly2))
                + F.dropout(self.fc3_u(fzly2)))

        ly4 = F.sigmoid(
                F.dropout(self.fc4_w(ly3))
                + F.dropout(self.fc4_u(fzly3)))

        return F.softmax(
                self.fc_out_w(ly4)
                + self.fc_out_u(fzly4))

##### train.py

# on pytorch

@curry
def validate_loop_lazy(name, __validate, loader):

    losses = [0.0] * len(loader)
    scores = [0.0] * len(loader)

    for i, batch in enumerate(loader):

        losses[i], scores[i]= __validate(batch)

    if len(loader) > 1:
        score = sum(scores[:-1])/(len(scores[:-1]))
        loss = sum(losses[:-1])/(len(losses[:-1]))

    else:
        score = scores[0]
        loss = losses[0]

    print('[%s] score: %.3f, loss: %.3f'
            %(name, score, loss))
        
    return loss, score


def train(model, loader, _valid_lazy, valid_loop, crit, optim):

    best_valid_score = 0.0
    best_model = model

    for epoch in range(ephs):
        for i, batch in enumerate(loader):

            inputs = batch[0]
            targets = batch[1]
            #print('batch',batch)

            optim.zero_grad()
            model.train() # autograd on

            train_loss = crit(model(inputs), targets)
            train_loss.backward()

            optim.step()
            model.eval() # autograd off

            __val_lz = _valid_lazy(model=model)

        print('[train] %5dth epoch, loss: %.3f'
                %(epoch, train_loss.data[0]))

        valid_loss, valid_score = valid_loop(__validate=__val_lz)

        if valid_score > best_valid_score:

            best_valid_score = valid_score

            print('[valid] bestscore: %.3f, loss: %.3f'
            %(valid_score, valid_loss))

            best_model = model

    print('Finished Training')

    return best_model

##### main.py

with open(pretrain_pk,'rb') as f:
    pretrainmat = pk.load(f)

with open(train_pk,'rb') as f:
    trainmat = pk.load(f)

dnn_y = np.int_(np.round(pretrainmat[:,0]))
dnn_cls = list(set(dnn_y))

prognet_y = np.int_(np.round(trainmat[:,0]))
prognet_cls = list(set(prognet_y))

dnn_cls_wgt = compute_class_weight('balanced', dnn_cls, dnn_y)
prognet_cls_wgt = compute_class_weight('balanced', prognet_cls, prognet_y)

dnn_cls_wgt = torch.FloatTensor(dnn_cls_wgt).to(device)
prognet_cls_wgt = torch.FloatTensor(prognet_cls_wgt).to(device)

print('dnn_cls_wgt', dnn_cls_wgt)
print('prognet_cls_wgt',prognet_cls_wgt)

valid_lazy = {'uar': validate_uar_lazy,
                'war': validate_war_lazy }

# loading

pretrainloader = DataLoader(egemaps_dataset(pretrain_pk, dnn_cls_wgt), bsz)
predevloader = DataLoader(egemaps_dataset(predev_pk, dnn_cls_wgt), bsz)
trainloader = DataLoader(egemaps_dataset(train_pk, prognet_cls_wgt), bsz)
devloader = DataLoader(egemaps_dataset(dev_pk, prognet_cls_wgt), bsz)
evalloader = DataLoader(egemaps_dataset(eval_pk, prognet_cls_wgt), bsz)

# pretraining

dnn_mdl = dnn()
dnn_mdl.to(device)

if os.path.exists(dnn_pth):
    print(dnn_pth, 'already exists.')
else:

    optim = torch.optim.Adam(dnn_mdl.parameters())

    precrit = nn.CrossEntropyLoss(weight=dnn_cls_wgt)

    _val_lz = valid_lazy[measure](crit=precrit)
    _val_loop_lz = validate_loop_lazy(name='valid', loader=predevloader)

    pretrained = train( dnn_mdl, pretrainloader, _val_lz, _val_loop_lz, precrit, optim )

    torch.save(pretrained.state_dict(), dnn_pth)


# training

#print('torch.load(dnn_pth)',torch.load(dnn_pth))
#print('dnn_mdl',dnn_mdl)
dnn_mdl.load_state_dict(torch.load(dnn_pth))

prog_mdl = prognet(dnn_mdl)
prog_mdl.to(device)
crit = nn.CrossEntropyLoss(weight=prognet_cls_wgt)
if os.path.exists(prognet_pth):
    print(prognet_pth, 'already exists')
else:

    optim = torch.optim.Adam(prog_mdl.parameters(), lr=0.00005)


    _val_lz = valid_lazy[measure](crit=crit)
    _val_loop_lz = validate_loop_lazy(name='valid', loader=devloader)

    trained = train(prog_mdl, trainloader, _val_lz, _val_loop_lz, crit, optim)

    torch.save(trained.state_dict(), prognet_pth)


prog_mdl.load_state_dict(torch.load(prognet_pth))
_val_lz = valid_lazy[measure](model=prog_mdl, crit=crit)

validate_loop_lazy('test', _val_lz, evalloader) 

