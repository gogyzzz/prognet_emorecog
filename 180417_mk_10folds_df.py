n_folds = 10
incre = len(valid_df)//n_folds
runs = []
for j in range(n_folds):
    folds = []
    splited = []
    shuffled = list(df.sample(len(df)).index)
    
    for i in range(n_folds):
        splited.append(shuffled[incre*(i):incre*(i+1)])
       
    for k in range(n_folds):
        test = splited[0]
        val = splited[1]
        train = list(set(shuffled) - set(test) - set(val))
        folds.append({'test':test, 'val':val, 'train':train})

    runs.append(folds)
    