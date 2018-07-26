import sys
from torch.utils.data import DataLoader

from egemaps_dataset import egemaps_dataset

# arg parsing

expcase = sys.argv[1]
#on_cuda = sys.argv[2]
on_cuda = True

#read %s/param.json
import json
param = {}
with open(expcase +'/param.json') as f:
    param = json.load(f)

#make dataset
pretrainloader = DataLoader(
        egemaps_dataset(param['dataset']+'/pretrain.pk', on_cuda), 
        param['bsz'])

trainloader = DataLoader(
        egemaps_dataset(param['dataset']+'/train.pk', on_cuda), 
        param['bsz'])

predevloader = DataLoader(
        egemaps_dataset(param['dataset']+'/predev.pk', on_cuda), 
        param['bsz'])

devloader = DataLoader(
        egemaps_dataset(param['dataset']+'/dev.pk', on_cuda), 
        param['bsz'])

evalloader = DataLoader(
        egemaps_dataset(param['dataset']+'/eval.pk', on_cuda), 
        param['bsz'])

# pretraining

nin = 88
nhid = 256
nout = 2
dnn_mdl = dnn(nin, nhid, nout, on_cuda)
optim = torch.optim.Adam(dnn_mdl.parameters())


if param['measure'] == 'war':
    score_func = partial(recall_score, average='weighted', 
            sample_weight=param['sample_weight'])

elif param['measure'] == 'uar':
    score_func = partial(recall_score, average='macro')

weight = [1.0, 1.0] 
precrit = nn.CrossEntropyLoss(weight=weight)

valid_func = partial(validate, 
        devloader=predevloader, crit=precrit, score_func=score_func,
        logpath=param['log'])
pretrained = train(pretrainloader, predevloader, dnn_mdl, precrit, 
        score_func, optim,
        param['lr'], param['pre_ephs'], param['log'])

torch.save(pretrained, param['premodel'])


# training

pretrained = torch.load_state_dict(torch.load(param['premodel']))

nout = 4
prognet_mdl = prognet(pretrained, nin, nhid, nout, on_cuda)
optim = torch.optim.Adam(prognet_mdl.parameters(), lr=0.00005)

weight = [1.0, 1.0] 
crit = nn.CrossEntropyLoss(weight=weight)

valid_func = partial(validate, 
        devloader=devloader, crit=crit, score_func=score_func,
        logpath=param['log'])
trained = train(trainloader, devloader, prognet_mdl, crit, 
        score_func, optim, 
        param['lr'], param['ephs'], param['log'])

torch.save(trained, param['model'])

test(testloader, trained, crit, score_func, 
    param['log'])
