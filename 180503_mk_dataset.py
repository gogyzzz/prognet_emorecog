# coding: utf-8
import pandas as pd

sessions = []

spk_pair = [('01F','01M'), ('02F','02M'), ('03F','03M'), ('04F','04M'), ('05F','05M')]

df = pd.read_hdf('iemocap_4emo.h5','table')

for isess in range(5):

    testdf = df.loc[df.loc[:,'speaker'].isin(spk_pair[0])]

    trainvaldf = df.loc[~df.loc[:,'speaker'].isin(spk_pair[0])]

    valdf = pd.concat([trainvaldf.loc[trainvaldf.loc[:,'cat'] == 'N'].sample(frac=0.1),
                        trainvaldf.loc[trainvaldf.loc[:,'cat'] == 'A'].sample(frac=0.1),
                        trainvaldf.loc[trainvaldf.loc[:,'cat'] == 'S'].sample(frac=0.1),
                        trainvaldf.loc[trainvaldf.loc[:,'cat'] == 'H'].sample(frac=0.1)],
                        ignore_index=True)
    
    traindf = trainvaldf.loc[~trainvaldf.loc[:,'id'].isin(valdf.loc[:,'id'])]    

    sessions.append((traindf, testdf, valdf))
    
#    print(testdf.head())
#    print(len(trainvaldf), len(valdf), len(traindf))
#    print(valdf.head())
#    print(traindf.head())
    
    #break
import pickle
pickle.dump(sessions, open('iemocap_5sessions_for_cv_frac_0.1.df.pk', 'wb'))
