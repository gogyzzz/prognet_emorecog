##### main.py

pretrainloader = DataLoader(pretrain_pk, bsz)
trainloader = DataLoader(train_pk, bsz)
predevloader = DataLoader(predev_pk, bsz)
devloader = DataLoader(dev_pk, bsz)
bevalloader = DataLoader(eval_pk, bsz)

# pretraining

dnn_mdl = dnn()

optim = torch.optim.Adam(dnn_mdl.parameters())

crit = nn.CrossEntropyLoss(weight=dnn_cls_wgt)

valid_func = partial(validate, 
        name='valid',loader=predevloader, crit=precrit)

pretrained = train(dnn_mdl, pretrainloader, valid_func, crit, optim)

torch.save(pretrained.state_dict, pretrain_pk)

# training

pretrained = torch.load_state_dict(torch.load(pretrain_pk))
prognet_mdl = prognet(pretrained)

optim = torch.optim.Adam(prognet_mdl.parameters(), lr=0.00005)

crit = nn.CrossEntropyLoss(weight=prognet_cls_wgt)

valid_func = partial(validate, 
        name='valid',loader=devloader, crit=crit)

trained = train(prognet_mdl, trainloader, valid_func, crit, optim)

torch.save(trained.state_dict, train_pk)

validate(trained, 'test', evalloader, crit=crit) 
