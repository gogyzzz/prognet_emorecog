import os
opensmiledir='/home/haeyong/opensmile-2.3.0/'
opensmile_config_path = '/home/haeyong/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf'
i=0
id_feat_dic = {}
for i, row in df.iterrows():
    print(row.loc['path'])
    os.system('{execute_path} -C {config} -I {wavpath}'.format(
        execute_path = opensmiledir + '/SMILExtract',
        config = opensmile_config_path,
        wavpath = row['path']))
        
    os.system('copy-feats --htk-in=true scp:{opensmiledir}/output.sink.scp ark,t:{opensmiledir}/output.sink.ark'.format(opensmiledir=opensmiledir))
    id_feat_dic[row['id']] = [float(e) for e in open(opensmiledir + '/output.sink.ark').readlines()[1].split()[:-1]]
    #row['egemaps'] = [float(e) for e in open(opensmiledir + '/output.sink.ark').readlines()[1].split()[:-1]]
    #print(row)
    #break
#    if i < 2:
#        i+=1
#    else:
    break

        
        
        
        
        
    