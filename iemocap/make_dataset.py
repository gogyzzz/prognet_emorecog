#!/usr/bin/env python

import os
import sys
import pandas as pd
import json as js
import numpy as np
import pickle as pk

csv = sys.argv[1] 
utt_egemaps_pk = sys.argv[2]
dataset = sys.argv[3]

# manual param.

devfrac=0.2
session=1
prelabel="gender"
featdim=88

# make 

full = pd.read_csv(csv)
print(full.head())
#print(full.columns)
#print(set(full.loc[:,'gender']))
prelabel_set = set(full.loc[:,prelabel])
label_set = set(full.loc[:,'emotion'])

idx_prelabel_dict = {i:label for i, label in enumerate(prelabel_set)}
idx_label_dict = {i:label for i, label in enumerate(label_set)}

os.system('mkdir -p '+dataset)

with open(dataset+'/idx_prelabel.json','w') as f:
    print(js.dumps(idx_prelabel_dict, sort_keys=True, indent=4), file=f)

with open(dataset+'/idx_label.json','w') as f:
    print(js.dumps(idx_label_dict, sort_keys=True, indent=4),file=f)

os.system('head %s/*.json'%(dataset))

# make dataset dataframe

prelabel_idx_dict = {label:i for i, label in enumerate(prelabel_set)}
label_idx_dict = {label:i for i, label in enumerate(label_set)}

_eval = full.loc[full.loc[:,'session'].isin([session])]
_traindev = full.loc[~full.loc[:,'session'].isin([session])]

_dev = pd.concat([_traindev.loc[_traindev.loc[:,'emotion'] == 'N'].sample(frac=devfrac),
    _traindev.loc[_traindev.loc[:,'emotion'] == 'A'].sample(frac=devfrac),
    _traindev.loc[_traindev.loc[:,'emotion'] == 'S'].sample(frac=devfrac),
    _traindev.loc[_traindev.loc[:,'emotion'] == 'H'].sample(frac=devfrac)],
    ignore_index=True)

_train = _traindev.loc[~_traindev.loc[:,'utterance'].isin(_dev.loc[:,'utterance'])]

print('')
print('number of samples fullset:',len(full))
print('number of samples traindev:',len(_traindev))
print('number of samples train:',len(_train))
print('number of samples dev:',len(_dev))
print('number of samples eval:',len(_eval))
print('')

# make dataset pk

utt_egemaps = pk.load(open(utt_egemaps_pk,'rb'))

pretrain_mat = np.ndarray(shape=(len(_train),1+featdim))
train_mat = np.ndarray(shape=(len(_train),1+featdim))
predev_mat = np.ndarray(shape=(len(_dev),1+featdim))
dev_mat = np.ndarray(shape=(len(_dev),1+featdim))
eval_mat = np.ndarray(shape=(len(_eval),1+featdim))

for i, irow in enumerate(_train.iterrows()):
    _, row = irow
    pretrain_mat[i,0] = prelabel_idx_dict[row[prelabel]]
    pretrain_mat[i,1:] = utt_egemaps[row['utterance']]
    train_mat[i,0] = label_idx_dict[row['emotion']]

for i, irow in enumerate(_dev.iterrows()):
    _, row = irow
    predev_mat[i,0] = prelabel_idx_dict[row[prelabel]]
    predev_mat[i,1:] = utt_egemaps[row['utterance']]
    dev_mat[i,0] = label_idx_dict[row['emotion']]

for i, irow in enumerate(_eval.iterrows()):
    _, row = irow
    eval_mat[i,0] = label_idx_dict[row['emotion']]
    eval_mat[i,1:] = utt_egemaps[row['utterance']]

# normalization

train_mean = np.mean(pretrain_mat[:,1:],axis=0)
train_std = np.std(pretrain_mat[:,1:],axis=0)

print('train_mean shape:',np.shape(train_mean))
print('train_std shape:',np.shape(train_std))

pretrain_mat[:,1:] = (pretrain_mat[:,1:] - train_mean)/train_std
predev_mat[:,1:] = (predev_mat[:,1:] - train_mean)/train_std
eval_mat[:,1:] = (eval_mat[:,1:] - train_mean)/train_std

train_mat[:,1:] = pretrain_mat[:,1:]
dev_mat[:,1:] = dev_mat[:,1:]

print('')
print('all dataset normalized.')
print('')

with open(dataset+'/pretrain.pk','wb') as f:
    pk.dump(pretrain_mat,f)

with open(dataset+'/train.pk','wb') as f:
    pk.dump(train_mat,f)

with open(dataset+'/predev.pk','wb') as f:
    pk.dump(predev_mat,f)

with open(dataset+'/dev.pk','wb') as f:
    pk.dump(dev_mat,f)

with open(dataset+'/eval.pk','wb') as f:
    pk.dump(eval_mat,f)

print('')
print('<',dataset,'>')
os.system('ls '+dataset)












