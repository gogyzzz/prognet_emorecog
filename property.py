##### property.py

device='cuda:0'

pretrain_pk = 'expdir/pretrain.pk'
trainpk_pk    = 'expdir/train.pk'
predev_pk   = 'expdir/predev.pk'
devpk_pk      = 'expdir/dev.pk'
evalpk_pk     = 'expdir/eval.pk'

dnn_pth = 'expdir/premodel.pth'
prognet_pth    = 'expdir/model.pth'


lr=0.00005
preephs=150
ephs=300
bsz=64

dnn_cls={0:'Male',1:'Female'}
prognet_cls={0:'Happiness', 1:'Sadness', 2:'Neutral', 3:'Anger'}

nin=88
nhid=256
dnn_nout=len(dnn_cls)
prognet_nout=len(prognet_cls)

measure='war'

