from sklearn.metrics import recall_score

def measure(measure_type, weight=[1]):

    if measure_type == 'uar':
        return partial(
                recall_score, average='macro')
    elif measure_type == 'war':
        return patrial(
                recall_score, average='weighted')
    else:
        print('no such measure type')
        return 0



            

    
