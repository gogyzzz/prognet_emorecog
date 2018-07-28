#!/usr/bin/env python

import os
import sys
import json as js

dataset = sys.argv[1]
exp = sys.argv[2]

# manual param.

lr=0.00005
pre_ephs=200 # pre epochs
ephs=200 # epochs
bsz=64 # batch size
measure="war"
nin=88
nhid=256

# 

os.system('mkdir -p '+exp)

param = { 'dataset':dataset,
            'lr':lr,
            'pre_ephs':pre_ephs,
            'ephs':ephs,
            'bsz':bsz,
            'measure':measure,
            'log':exp+'/log',
            'premodel':exp+'/premodel.pth',
            'nin':nin,
            'nhid':nhid,
            'model':exp+'/model.pth' }


with open(exp+'/param.json','w') as f:
    js.dump(param,f, sort_keys=True, indent=4)
    print(js.dumps(param, sort_keys=True, indent=4))

print('')
print('<',exp,'>')
os.system('ls '+exp)
