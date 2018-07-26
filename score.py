

if measure == 'war':
    print('''

    smaple_weight = 
    score += recall_score(\
            outputs.max(dim=1)[1].data.cpu().numpy(),\
            targets.data.cpu().numpy()),
            average='weighted',
            sample_weight=sample_wgts)

    ''')

elif measure == 'uar':
    print('''

    smaple_weight = 
    score += recall_score(\
            outputs.max(dim=1)[1].data.cpu().numpy(),\
            targets.data.cpu().numpy()),
            average='macro')

    ''')
