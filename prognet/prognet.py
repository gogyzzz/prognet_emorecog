#xxx prognet.py

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def init_linear(m):
    if type(m) == nn.Linear:
        tc.nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)

class dnn(nn.Module):
    def __init__(self, nin, nhid, nout):
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

        self.fc_out = nn.Linear(nhid, nout)

        self.apply(init_linear)

    def forward(self, x):

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(F.dropout(self.fc2(x)))
        x = F.sigmoid(F.dropout(self.fc3(x)))
        x = F.sigmoid(F.dropout(self.fc4(x)))
        x = F.softmax(self.fc_out(x))

        return x

class prognet(nn.Module):

    def __init__(self, pretrained, nin, nhid, nout): 

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

        self.fc_out_w = nn.Linear(nhid, nout)
        self.fc_out_u = nn.Linear(nhid, nout) 

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

