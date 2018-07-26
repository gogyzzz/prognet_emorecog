##### egemaps_dataset.py

class egemaps_dataset(Dataset):

    def __init__(self, pkpath, cls_wgt):

        with open(pkpath, 'rb') as f:
            # (samples x dimension)
            print(pkpath, 'shape:', np.shape(datamat)) 

            datamat = pk.load(f)

            self.targets = Variable( 
                    torch.LongTensor(int(np.round(datamat[:,0])),
                        device=device)).view(-1)

            self.inputs = Variable( 
                    torch.FloatTensor(datamat[:,1:],
                        device=device))

            self.sample_wgt = [cls_wgt[i] for i in 
                    int(np.round(datamat[:,0]))]

            print('check sef of sample_weight:',set(sample_wgt))


    def __len__(self):
        return np.shape(dataset)[0]

    def __getitem__(self, idx):
        return (self.inputs[idx], self.targets[idx], self.sample_wgt[idx])
