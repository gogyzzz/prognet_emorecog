
n_folds = 10
incre = len(valid_df)//n_folds
folded_dfs = []
shuf_df = valid_df.sample(len(valid_df))
for i in range(n_folds):
    #print(incre*(i),'-',incre*(i+1))
    folded_dfs.append(shuf_df.iloc[incre*(i):incre*(i+1)])