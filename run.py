# coding: utf-8
# %load run.py
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# Dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import pickle as pk
import numpy as np
import torch.nn as nn

def to_cu(is_cuda, tensor):

    if is_cuda is True:
        return tensor.cuda()
    else:
        return tensor
    

class EgemapsDataset(Dataset):
    """DB Dataset."""
    
    def __init__(self, df, id_feat_dic_path, is_cuda):

        self.is_cuda = is_cuda

        # available label : speaker, gender, cat
        self.df = df

        self.id_feat_dic = pk.load(open(id_feat_dic_path,'rb'))

        for key, value in self.id_feat_dic.items():
            self.id_feat_dic[key] = to_cu(is_cuda, torch.FloatTensor(value))

        self.gender_digit_dict = {}
        self.gender_digit_dict['F'] = to_cu(is_cuda, torch.LongTensor([0]))
        self.gender_digit_dict['M'] = to_cu(is_cuda, torch.LongTensor([1]))

        spk_list = list(set(self.df.loc[:,'speaker']))

        self.speaker_digit_dict = {spk:i for i, spk in enumerate(spk_list)}

        for key, value in self.speaker_digit_dict.items():
            self.speaker_digit_dict[key] = to_cu(is_cuda, torch.LongTensor([value]))

#        for i,spk in enumerate(spk_list):
#            self.speaker_onehot[spk] = ([0]*len(spk_list))
#            self.speaker_onehot[spk][i] = 1

        cat_list = list(set(self.df.loc[:,'cat']))

        self.cat_digit_dict = {cat:i for i, cat in enumerate(cat_list)}

        for key, value in self.cat_digit_dict.items():
            self.cat_digit_dict[key] = to_cu(is_cuda, torch.LongTensor([value]))
#        for i, cat in enumerate(cat_list):
#            self.cat_onehot[cat] = ([0]*len(cat_list))
#            self.cat_onehot[cat][i] = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        egemaps = self.id_feat_dic[self.df.iloc[idx]['id']]
        gender = self.gender_digit_dict[self.df.iloc[idx]['gender']]
        speaker = self.speaker_digit_dict[self.df.iloc[idx]['speaker']]
        cat = self.cat_digit_dict[self.df.iloc[idx]['cat']]

        return {'egemaps':egemaps,'gender':gender,'speaker':speaker,'cat':cat}
        

class RCVEgemapsDataset(Dataset):
    """dataset for repeated cross validation"""
    def __init__(self, dataset, runs_path, irun=0, ifold=0, mode='train'):
        self.dataset = dataset
        self.runs = pk.load(open(runs_path, 'rb'))
        
        self.irun = irun
        self.ifold = ifold
        self.mode = mode


    
    def __len__(self):
        return len(self.runs[self.irun][self.ifold][self.mode])

    def __getitem__(self, idx):
        
        return self.dataset[self.runs[self.irun][self.ifold][self.mode][idx]]

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)
        

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class BasicDNN(nn.Module):

    def __init__(self, n_labels, mode, is_cuda):
        super(BasicDNN, self).__init__()

        self.n_lables = n_labels
        self.mode = mode # mode can be speaker, gender or cat
        self.is_cuda = is_cuda

        self.bn0 = nn.BatchNorm1d(88)

        self.fc1 = nn.Linear(88, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout()
        

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.do2 = nn.Dropout()
        
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.do3 = nn.Dropout()

        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.do4 = nn.Dropout()

        self.fc5 = nn.Linear(256, n_labels)
        # Three different targets
        #self.fc_speaker = nn.Linear(256, n_labels)
        #self.bn_speaker = nn.BatchNorm1d(n_labels)
        #self.do_speaker = nn.Dropout()

        #self.fc_gender = nn.Linear(256, n2)
        #self.bn_gender = nn.BatchNorm1d(2)
        #self.do_gender = nn.Dropout()

        #self.fc_cat = nn.Linear(256, 4)
        #self.bn_cat = nn.BatchNorm1d(4)
        self.do_cat = nn.Dropout()
        

        if self.is_cuda is True:
            self.cuda()

        self.apply(init_weights)

    def setmode(self, mode):
        self.mode = mode

    def getmode(self):
        return self.mode

    def forward(self, x):
        """
        mode can be speaker, gender and cat
        """

        #x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.fc2(x)))
        #x = F.relu(self.bn3(self.fc3(x)))
        #x = F.relu(self.bn4(self.fc4(x)))
        
        #x = self.bn0(x)
        #x = F.relu(F.dropout(self.fc1(x)))
        #x = F.relu(F.dropout(self.fc2(x)))
        #x = F.relu(F.dropout(self.fc3(x)))
        #x = F.relu(F.dropout(self.fc4(x)))

        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = F.relu(self.do3(self.fc3(x)))
        x = F.relu(self.do4(self.fc4(x)))
        return F.softmax(self.fc5(x))

        #if self.mode == 'speaker':
            #x = F.softmax(self.bn_speaker(self.fc_speaker(x)))
            #x = F.softmax(F.dropout(self.bn_speaker(self.fc_speaker(x))))
        #    x = F.softmax((self.fc_speaker(x)))
            #x = self.bn_speaker(self.fc_speaker(x))
        #    return x

        #elif self.mode == 'gender':
            #x = F.softmax(self.bn_gender(self.fc_gender(x)))
            #x = F.softmax(F.dropout(self.bn_gender(self.fc_gender(x))))
        #    x = F.softmax((self.fc_gender(x)))
            #x = self.bn_gender(self.fc_gender(x))
        #    return x

        #elif self.mode == 'cat':
            #x = F.softmax(self.bn_cat(fc_cat(x)))
            #x = F.softmax(F.dropout(self.bn_cat(fc_cat(x))))
        #    x = F.softmax((self.fc_cat(x)))
            #x = self.bn_cat(fc_cat(x))
        #    return x

        #else:
        #    print(' >> caution << : unvalid mode')
        #    return x


class ProgNet(nn.Module):

    def __init__(self, basicdnn_state_dict, num_classes, mode, is_cuda):
        super(ProgNet, self).__init__()

        self.basicdnn_state_dict = basicdnn_state_dict

        self.num_classes = num_classes 
        self.mode = mode # mode can be speaker, gender or cat
        self.is_cuda = is_cuda

        self.bn0 = nn.BatchNorm1d(88)

        self.fc1_w = nn.Linear(88, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout()

        self.fc2_w = nn.Linear(256, 256)
        self.fc2_u = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.do2 = nn.Dropout()
        
        self.fc3_w = nn.Linear(256, 256)
        self.fc3_u = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.do3 = nn.Dropout()

        self.fc4_w = nn.Linear(256, 256)
        self.fc4_u = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.do4 = nn.Dropout()

        # Three different targets
        self.fc_speaker_w = nn.Linear(256, num_classes)
        self.fc_speaker_u = nn.Linear(256, num_classes)
        self.bn_speaker = nn.BatchNorm1d(num_classes)
        self.do_speaker = nn.Dropout()

        self.fc_gender_w = nn.Linear(256, 2)
        self.fc_gender_u = nn.Linear(256, 2)
        self.bn_gender = nn.BatchNorm1d(2)
        self.do_gender = nn.Dropout()

        self.fc_cat_w = nn.Linear(256, num_classes)
        self.fc_cat_u = nn.Linear(256, num_classes)
        self.bn_cat = nn.BatchNorm1d(num_classes)
        self.do_cat = nn.Dropout()

        if self.is_cuda is True:
            self.cuda()

        self.apply(init_weights)

    def setmode(self, mode):
        self.mode = mode

    def getmode(self):
        return self.mode

    def get_layers(self, x):
#        fz_fc1 = self.basicdnn_state_dict['net_state_dict']['fc1.
        fz_fcw1 = Variable(self.basicdnn_state_dict['net_state_dict']['fc1.weight'])
        fz_fcb1 = Variable(self.basicdnn_state_dict['net_state_dict']['fc1.bias'])
        fz_fcw2 = Variable(self.basicdnn_state_dict['net_state_dict']['fc2.weight'])
        fz_fcb2 = Variable(self.basicdnn_state_dict['net_state_dict']['fc2.bias'])
        fz_fcw3 = Variable(self.basicdnn_state_dict['net_state_dict']['fc3.weight'])
        fz_fcb3 = Variable(self.basicdnn_state_dict['net_state_dict']['fc3.bias'])
        fz_fcw4 = Variable(self.basicdnn_state_dict['net_state_dict']['fc4.weight'])
        fz_fcb4 = Variable(self.basicdnn_state_dict['net_state_dict']['fc4.bias'])
        
        fz_fcw1.requires_grad = False
        fz_fcb1.requires_grad = False
        fz_fcw2.requires_grad = False
        fz_fcb2.requires_grad = False
        fz_fcw3.requires_grad = False
        fz_fcb3.requires_grad = False
        fz_fcw4.requires_grad = False
        fz_fcb4.requires_grad = False

        #fzly1 = F.relu(F.linear(x, fz_fcw1, fz_fcb1))
        #fzly2 = F.relu(F.linear(fzly1, fz_fcw2, fz_fcb2))
        #fzly3 = F.relu(F.linear(fzly2, fz_fcw3, fz_fcb3))
        #fzly4 = F.relu(F.linear(fzly3, fz_fcw4, fz_fcb4))
        
        fzly1 = F.relu((F.linear(x, fz_fcw1, fz_fcb1)))
        fzly2 = F.relu((F.linear(fzly1, fz_fcw2, fz_fcb2)))
        fzly3 = F.relu((F.linear(fzly2, fz_fcw3, fz_fcb3)))
        fzly4 = F.relu((F.linear(fzly3, fz_fcw4, fz_fcb4)))

        return fzly1, fzly2, fzly3, fzly4

    def forward(self, x):
        """
        mode can be speaker, gender and cat
        """

        fzly1, fzly2, fzly3, fzly4 = self.get_layers(x)

        ly1 = F.relu(self.do1(self.fc1_w(x)))
        ly2 = F.relu(self.do2(self.fc2_w(ly1) + self.fc2_u(fzly1)))
        ly3 = F.relu(self.do3(self.fc3_w(ly2) + self.fc3_u(fzly2)))
        ly4 = F.relu(self.do4(self.fc4_w(ly3) + self.fc4_u(fzly3)))
        #print(self.training)
        #if self.training == True:
            #print('training = True')
            #print(ly4.data[0:2,0:2].cpu())
        #else:
        #    print('training = False')
        #    print(ly4.data[0:2,0:2].cpu())

        if self.mode == 'speaker':
            return F.softmax(
                    (self.fc_speaker_w(ly4)
                    + self.fc_speaker_u(fzly4)))

        elif self.mode == 'gender':
            return F.softmax(
                    (self.fc_gender_w(ly4)
                    + self.fc_gender_u(fzly4)))

        elif self.mode == 'cat':
             return F.softmax(
                    (self.fc_cat_w(ly4)
                    + self.fc_cat_u(fzly4)))

        else:
            print(' >> caution << : unvalid mode')
            return x




import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import recall_score # for validate
from torch.utils.data import DataLoader

import os


def validate(validset, network, criterion, mode):

    dataloader = DataLoader(validset, batch_size=len(validset)) # get full size batch

    for batch in dataloader: # 1 iter
    #batch = dataloader.next() # get full size batch

        x_input = Variable(batch['egemaps'])
        y_true = Variable(batch[mode]).view(-1)

        #print('before eval:',network.training)
        #network.eval()
        #print('after eval:',network.training)


        output = network(x_input)
        loss = criterion(output, y_true)
#        print(output.max(1))

        y_pred = output.max(dim=1)[1]

        uar = recall_score(
                y_pred.data.cpu().numpy(),
                y_true.data.cpu().numpy(),
                    average='macro')

#        print(y_pred.data.cpu().numpy()[:10])
#        print(y_true.data.cpu().numpy()[:10])



    return loss, uar



class Trainer():

    def __init__(self, modelpath, traindataloader, validset, testset, network, mode, st_epoch, ed_epoch, optimizer, criterion, is_cuda):

        self.modelpath = modelpath

        self.traindataloader = traindataloader

        self.validset = validset

        self.testset = testset
        
        self.network = network

        if mode == network.mode:
            self.mode = mode
        else:
            print('error: mode != network.mode')

        self.st_epoch = st_epoch
        self.ed_epoch = ed_epoch
        
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_losses = []
        self.val_uars = []


    def training(self):

        best_valid_uar = 0.0

        for epoch in range(self.st_epoch, self.ed_epoch):

            for i, batch in enumerate(self.traindataloader):

                x_inputs = Variable(batch['egemaps'])

                y_labels = Variable(batch[self.mode]).view(-1)

                self.optimizer.zero_grad()
#                print(self.network.forward(x_inputs))
#                print(y_labels)

#                print(x_inputs)
#                print(y_labels)
#                loss = F.cross_entropy(self.network.forward(x_inputs), y_labels)

                self.network.train()

                forwarded = self.network.forward(x_inputs)
                #print('training')
                train_loss = self.criterion(forwarded, y_labels)

                #print(train_loss.data[0])
                train_loss.backward()

                self.optimizer.step()

#                print('fc1.weight:',self.network.fc1.weight)
                # print('fc4.weight:',self.network.fc4.weight)
                # print('forwarded:',forwarded)
            
                #self.network.train(False)

            self.network.eval()
            valid_loss, valid_uar = validate(
                    self.validset, self.network, 
                    self.criterion, self.mode)

            train_log = '[%d, %5d] loss: %.3f'%(epoch, i, train_loss.data[0])
            valid_log = '[%d, %5d] valid_uar: %s'%(epoch, i, str(valid_uar))
            os.system('echo %s >> %s.log' %(train_log, self.modelpath))
            os.system('echo %s >> %s.log' %(valid_log, self.modelpath))
            print(train_log)


            if valid_uar > best_valid_uar:

                best_valid_uar = valid_uar
                state = {
                        'net_state_dict':self.network.state_dict(),
                        'train_loss':train_loss,
                        'valid_uar':valid_uar,
                        'optim_state_dict':self.optimizer.state_dict()}
                # filename = '_'.join([self.modelpath, str(epoch), '.pth'])

                valid_log = '[%d, %5d] best valid_uar: %s'%(epoch, i, str(best_valid_uar))
                os.system('echo %s >> %s.log' %(valid_log, self.modelpath))
                print(valid_log)

        torch.save(state, self.modelpath)
        print('save model ' + self.modelpath)
        print('Finished Training')                      

        return self.modelpath

    def test(self):
        self.network.eval()
        test_loss, test_uar = validate(
            self.testset, self.network, 
            self.criterion, self.mode)
        
        test_log = 'test_uar: %s'%(str(test_uar))
        os.system('echo %s >> %s.log' %(test_log, self.modelpath))
        print(test_log)


import pickle as pk
def run_exp_case(exp_case):

    #is_cuda = True
    is_cuda = False

    exp_case = json.loads(exp_case)
    
    datasets = pk.load(open(exp_case['dataset_pk'],'rb'))

    #if exp_case['db_hdf'] == 'iemocap_4emo.h5':
    #    catweight = to_cu(is_cuda, Variable(torch.Tensor(pk.load(open('iemocap_balanced_weights.pk','rb')))))
    #else:
    #    catweight = to_cu(is_cuda, Variable(torch.Tensor(pk.load(open('msp_improv_balanced_weights.pk','rb')))))

#    egemapsdataset = EgemapsDataset(exp_case['db_hdf'],exp_case['id_feat_dic_pk'], is_cuda)
    
    trainset = EgemapsDataset(datasets['train'], exp_case['id_feat_dic_pk'], is_cuda)

    traindataloader = DataLoader(trainset, batch_size=exp_case['batch_size'], shuffle=True)

    validset = EgemapsDataset(datasets['valid'], exp_case['id_feat_dic_pk'], is_cuda)

    testset = EgemapsDataset(datasets['test'], exp_case['id_feat_dic_pk'], is_cuda)

    #if exp_case['pre_label'] == 'cat':
    #    criterion = nn.CrossEntropyLoss(weight=catweight)

    #else:
        #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    model = BasicDNN(exp_case['n_prelabels'],exp_case['pre_label'], is_cuda)

    #optimizer = torch.optim.Adam(model.parameters())
    #optimizer = torch.optim.SGD(model.parameters(), lr=exp_case['lr'], momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp_case['lr'])

    trainer = Trainer(exp_case['best_premodel_path'],traindataloader, validset, testset, model, exp_case['pre_label'], 0, exp_case['n_pre_epochs'], optimizer, criterion, is_cuda)

    trainer.training()
    trainer.test()

    prev_state_dict = torch.load(exp_case['best_premodel_path'])

    model = ProgNet(prev_state_dict, exp_case['n_classes'], 'cat', is_cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=exp_case['lr'])
    #optimizer = torch.optim.SGD(model.parameters(), lr=exp_case['lr'], momentum=0.9)

    traindataloader = DataLoader(trainset, batch_size=exp_case['batch_size'], shuffle=True)

    trainer = Trainer(exp_case['best_prognet_path'],traindataloader, validset, testset, model, 'cat', 0, exp_case['n_epochs'], optimizer, criterion, is_cuda)

    print('--- Training Progressive neural network ---')
    trainer.training()

    trainer.test()



import json
import sys

exp_cases = open(sys.argv[1]).readlines()
for exp_case in exp_cases:
    
    print(exp_case)
    run_exp_case(exp_case)
    #break


