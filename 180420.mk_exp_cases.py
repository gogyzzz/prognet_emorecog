
import json
import itertools as it

date='180420'
db_names = ['iemocap','msp_improv']
dbs = ['iemocap_4emo.h5','msp_improv_valid.h5']
id_feat_dics = ['iemocap_id_normed_feat_dic.pk','msp_improv_id_normed_feat_dic.pk']
runs = ['iemocap_runs.pk','msp_improv_valid_runs.pk']
n_runs = 1
n_folds = 10
n_spks = [10, 12]
n_cats = 4

pre_labels = ['speaker', 'gender', 'cat']
pre_n_labels = [[10, 2, 4], [12, 2, 4]]

f = open('180420.exp_cases','w')

for idb, irun, ifold, ilabel in list(it.product(range(2),range(n_runs),range(n_folds),range(3))):
    
    if (idb == 0) and (ifold > 7 or irun > 7):
        continue

    print('''\n
    database: {db}\n
    id_feat_dic: {id_feat_dic}\n
    run: {irun}\n
    run_file: {run_file}\n
    fold: {ifold}\n
    n_spk: {n_spk}\n
    label: {label}\n'''.format(
    db=dbs[idb], id_feat_dic=id_feat_dics[idb],
    irun=irun, run_file=runs[idb], 
    ifold=ifold, n_spk=n_spks[idb], label=pre_labels[ilabel]))
    
    premodel_name = '{date}_{db_name}_run{irun}{ifold}_{pre_label}'.format(
        date=date,
        db_name=db_names[idb],
        irun=irun, ifold=ifold,
        pre_label=pre_labels[ilabel])
        
    prognet_name = premodel_name + '2cat'
    
    exp_case = {
        'db_hdf':dbs[idb],
        'id_feat_dic_pk':id_feat_dics[idb],
        'run_fold_pk':runs[idb],
        'irun':irun, 'ifold':ifold,
        'pre_label':pre_labels[ilabel],
        'n_labels':pre_n_labels[idb][ilabel],
        'premodel_name':premodel_name,
        'prognet_name':prognet_name
        }
    #os.system('echo %s >> 180420.exp_cases'%(json.dumps(exp_case)))
    print(json.dumps(exp_case),file=f)
    #break
f.close()

    
