##### class_weight.py

with open(pretrain_pk) as f:
    pretrainmat = pk.load(f)

with open(train_pk) as f:
    trainmat = pk.load(f)

dnn_y = int(np.round(pretrainmat[:,0]))
dnn_cls = set(dnn_y)

prognet_y = int(np.round(trainmat[:,0]))
prognet_cls = set(prognet_y)

print('dnn_cls_wgt = ',compute_class_weight('balanced', dnn_cls, dnn_y))
print('prognet_cls_wgt = ',compute_class_weight('balanced', prognet_cls, prognet_y))
