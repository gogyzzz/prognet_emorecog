# coding: utf-8
import pandas as pd

sessions = []

df = pd.read_hdf('iemocap_4emo.h5','table')

for isess in range(5):

    testdf = df.loc[df.loc[:,'speaker'].isin(spk_pair[0])]

    trainvaldf = df.loc[~df.loc[:,'speaker'].isin(spk_pair[0])]

    valdf = pd.concat([trainvaldf.loc[trainvaldf.loc[:,'cat'] == 'N'].sample(frac=0.2),
                        trainvaldf.loc[trainvaldf.loc[:,'cat'] == 'A'].sample(frac=0.2),
                        trainvaldf.loc[trainvaldf.loc[:,'cat'] == 'S'].sample(frac=0.2),
                        trainvaldf.loc[trainvaldf.loc[:,'cat'] == 'H'].sample(frac=0.2)],
                        ignore_index=True)
    
    traindf = trainvaldf.loc[~trainvaldf.loc[:,'id'].isin(valdf.loc[:,'id'])]    

    sessions.append((traindf, testdf, valdf))
    
#    print(testdf.head())
#    print(len(trainvaldf), len(valdf), len(traindf))
#    print(valdf.head())
#    print(traindf.head())
    
    #break
import pickle
pickle.dump(sessions, open('iemocap_5sessions_for_cv.df.pk', 'wb'))
