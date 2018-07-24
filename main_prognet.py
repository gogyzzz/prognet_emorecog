import sys

# arg parsing

expcase = sys.argv[1]
on_cuda = sys.argv[2]

#read %s/param.json
import json
param = {}
with open(expcase +'/param.json') as f:
    param = json.load(f)

#make dataset
pretrainloader = egemaps_dataloader(param['dataset']+'/pretrain.pk', param['bsz'], on_cuda)
trainloader = egemaps_dataloader(param['dataset']+'/train.pk', param['bsz'], on_cuda)
predevloader = egemaps_dataloader(param['dataset']+'/predev.pk', param['bsz'], on_cuda)
devloader = egemaps_dataloader(param['dataset']+'/dev.pk', param['bsz'], on_cuda)
evalloader = egemaps_dataloader(param['dataset']+'/eval.pk', param['bsz'], on_cuda)

dnn_mdl = dnn(param['premodel'], on_cuda)
prognet_mdl = prognet(param['model'], on_cuda)

score_func = measure(param['measure'], on_cuda) # war or uar
train(dnn_mdl, score_func, param['lr'], param['pre_ephs'], param['log'])
train(prognet_mdl, score_func, param['lr'], param['ephs'], param['log'])
test(prognet_mdl, score_func, param['log'])
