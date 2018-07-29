Progressive Neural Networks for Transfer Learning in Emotion Recognition
===


[paper](https://arxiv.org/pdf/1706.03256.pdf)

[IEMOCAP DB paper](https://pdfs.semanticscholar.org/5cf0/d213f3253cd46673d955209f8463db73cc51.pdf)

[MSP-IMPROV DB paper](https://web.eecs.umich.edu/~emilykmp/EmilyPapers/2016_Busso_TAFF.pdf)

## Requirements

- Python 3.6.4

- Pytorch 0.4.1

- [opensmile 2.3.0](https://www.audeering.com/research/opensmile)

- [fileutils.readHtk](https://github.com/MaigoAkisame/fileutils)(githubrepo. I changed htk.py for python3)




## Preparation

### wav_cat.list, utt.list

IEMOCAP DB has 5531 utterances, composed of 4 Emotions.

A: Anger
H: Excited + Happiness
N: Neutral
S: Sadness

```bash
#head -2 iemocap/wav_cat.list
/your/path/Ses01F_impro01_F000.wav N
/your/path/Ses01F_impro01_F001.wav N

#head -2 iemocap/utt.list
Ses01F_impro01_F000
Ses01F_impro01_F001

```

MSP-IMPROV DB has 7798 utterances, composed of 4 Emotions.

```bash
#head -2 msp_improv/wav_cat.list
/your/path/MSP-IMPROV-S01A-F01-P-FM01.wav N 
/your/path/MSP-IMPROV-S01A-F01-P-FM02.wav H

#head -2 msp_improv/utt.list
MSP-IMPROV-S01A-F01-P-FM01
MSP-IMPROV-S01A-F01-P-FM02
```

## How to Run

```bash
./add_opensmile_conf.sh your_opensmile_dir

./prepare_list.sh iemocap/wav_cat.list \
	iemocap/egemaps.htk.list iemocap/utt.list iemocap/egemaps/

./extract_egemaps.sh your_opensmile_dir/ iemocap/wav_cat.list \
	iemocap/egemaps.htk.list

./make_utt_egemaps_pair.py iemocap/utt.list iemocap/egemaps.htk.list \
	iemocap/utt_egemaps.pk

./iemocap/make_csv.sh iemocap/utt.list iemocap/wav_cat.list iemocap/ \
	iemocap/full_dataset.csv

# Modify make_dataset.py parameters as you want!
#
### Default setting ###
#
# devfrac=0.2
# session=1
# prelabel="gender"
#
# e.g.
# sed 's/"gender"/"speaker"/' iemocap/make_dataset.py > new_script.py
# sed 's/devfrac=0.2/devfrac=0.1/' iemocap/make_dataset.py > new_script.py

./iemocap/make_dataset.py iemocap/full_dataset.csv iemocap/utt_egemaps.pk iemocap/your_dataset_path

# Modify make_expcase.py params as you want!
#
### Default setting ###
#
# lr=0.00005
# bsz=64
# ephs=200

./iemocap/make_expcase.py iemocap/your_dataset_path iemocap/your_dataset_path/your_expcase

ls iemocap/your_dataset_path/your_expcase 

# log	
# param.json
# premodel.pth
# model.pth

./run.py --propjs iemocap/your_dataset_path/your_expcase/param.json \
	> iemocap/your_dataset_path/your_expcase/log

grep test iemocap/your_dataset_path/your_expcase/log

# exp results ( Gender => Emotion case )

iemocap/sess11/exp/log:[test] score: 0.503, loss: 1.219
iemocap/sess12/exp/log:[test] score: 0.507, loss: 1.220
iemocap/sess13/exp/log:[test] score: 0.504, loss: 1.218
iemocap/sess14/exp/log:[test] score: 0.504, loss: 1.220
iemocap/sess15/exp/log:[test] score: 0.501, loss: 1.220
iemocap/sess21/exp/log:[test] score: 0.547, loss: 1.173
iemocap/sess22/exp/log:[test] score: 0.551, loss: 1.178
iemocap/sess23/exp/log:[test] score: 0.547, loss: 1.176
iemocap/sess24/exp/log:[test] score: 0.548, loss: 1.173
iemocap/sess25/exp/log:[test] score: 0.546, loss: 1.180
iemocap/sess31/exp/log:[test] score: 0.550, loss: 1.177
iemocap/sess32/exp/log:[test] score: 0.547, loss: 1.179
iemocap/sess33/exp/log:[test] score: 0.547, loss: 1.181
iemocap/sess34/exp/log:[test] score: 0.538, loss: 1.184
iemocap/sess35/exp/log:[test] score: 0.550, loss: 1.180
iemocap/sess41/exp/log:[test] score: 0.548, loss: 1.174
iemocap/sess42/exp/log:[test] score: 0.553, loss: 1.173
iemocap/sess43/exp/log:[test] score: 0.555, loss: 1.176
iemocap/sess44/exp/log:[test] score: 0.552, loss: 1.175
iemocap/sess45/exp/log:[test] score: 0.542, loss: 1.176
iemocap/sess51/exp/log:[test] score: 0.510, loss: 1.206
iemocap/sess52/exp/log:[test] score: 0.510, loss: 1.211
iemocap/sess53/exp/log:[test] score: 0.505, loss: 1.212
iemocap/sess54/exp/log:[test] score: 0.502, loss: 1.216
iemocap/sess55/exp/log:[test] score: 0.525, loss: 1.200

msp_improv/sess11/exp/log:[test] score: 0.446, loss: 1.267
msp_improv/sess12/exp/log:[test] score: 0.464, loss: 1.263
msp_improv/sess13/exp/log:[test] score: 0.451, loss: 1.264
msp_improv/sess14/exp/log:[test] score: 0.464, loss: 1.254
msp_improv/sess15/exp/log:[test] score: 0.453, loss: 1.265
msp_improv/sess21/exp/log:[test] score: 0.437, loss: 1.279
msp_improv/sess22/exp/log:[test] score: 0.435, loss: 1.276
msp_improv/sess23/exp/log:[test] score: 0.432, loss: 1.284
msp_improv/sess24/exp/log:[test] score: 0.444, loss: 1.269
msp_improv/sess25/exp/log:[test] score: 0.443, loss: 1.282
msp_improv/sess31/exp/log:[test] score: 0.461, loss: 1.262
msp_improv/sess32/exp/log:[test] score: 0.454, loss: 1.267
msp_improv/sess33/exp/log:[test] score: 0.458, loss: 1.266
msp_improv/sess34/exp/log:[test] score: 0.454, loss: 1.265
msp_improv/sess35/exp/log:[test] score: 0.453, loss: 1.273
msp_improv/sess41/exp/log:[test] score: 0.470, loss: 1.252
msp_improv/sess42/exp/log:[test] score: 0.458, loss: 1.258
msp_improv/sess43/exp/log:[test] score: 0.453, loss: 1.260
msp_improv/sess44/exp/log:[test] score: 0.464, loss: 1.258
msp_improv/sess45/exp/log:[test] score: 0.471, loss: 1.253
msp_improv/sess51/exp/log:[test] score: 0.541, loss: 1.193
msp_improv/sess52/exp/log:[test] score: 0.516, loss: 1.212
msp_improv/sess53/exp/log:[test] score: 0.529, loss: 1.201
msp_improv/sess54/exp/log:[test] score: 0.531, loss: 1.197
msp_improv/sess55/exp/log:[test] score: 0.516, loss: 1.211
msp_improv/sess61/exp/log:[test] score: 0.440, loss: 1.284
msp_improv/sess62/exp/log:[test] score: 0.471, loss: 1.263
msp_improv/sess63/exp/log:[test] score: 0.466, loss: 1.259
msp_improv/sess64/exp/log:[test] score: 0.456, loss: 1.274
msp_improv/sess65/exp/log:[test] score: 0.444, loss: 1.276
```
