Progressive Neural Network for Speech Emotion Recognition
===

## Requirements

- Python 3.6.4

- Pytorch 0.4.0 

- [csvkit](https://csvkit.readthedocs.io)

- [opensmile 2.3.0](https://www.audeering.com/research/opensmile)

## Preparation

### Wav-Categorical_Emotion Pair List File

Wav list file for IEMOCAP DB has 5531 utterances, composed of 4 Emotions.

A: Anger
H: Excited + Happiness
N: Neutral
S: Sadness

```bash
#head -2 iemocap/wav_cat.list
/your/path/Ses01F_impro01_F000.wav N
/your/path/Ses01F_impro01_F001.wav N
...
```

Wav list file for MSP-IMPROV DB has 7798 utterances, composed of 4 Emotions.

```bash
#head -3 msp_improv/wav_cat.list
/your/path/MSP-IMPROV-S01A-F01-P-FM01.wav N 
/your/path/MSP-IMPROV-S01A-F01-P-FM02.wav H
/your/path/MSP-IMPROV-S01A-F01-P-MF01.wav H
...
```

## How to Run

### IEMOCAP DB

```bash
iemocap/make_csv.sh iemocap/wav_cat.list iemocap/

```

```bash
# head iemocap/db.csv | csvlook ls iemocap 
wav_cat.list 
db.csv
wav_egemaps.htk.list
egemaps_cat.htk.list
egemaps/
```

```bash
cat iemocap/wav_cat.list | parallel --colsep ' ' bash ./extract_egemaps.sh {}

# read egemaps convert numpy matrix pickle

convert_egemaps_cat_to_np_matrix.py


iemocap/egemaps.pk

# make 10 cross validation datasets

python 

iemocap/dataset/run/0/fold/0/train/egemaps.pk
iemocap/dataset/run/0/fold/0/dev/egemaps.pk
iemocap/dataset/run/0/fold/0/eval/egemaps.pk
...
iemocap/dataset/run/9/fold/9/train/egemaps.pk
iemocap/dataset/run/9/fold/9/dev/egemaps.pk
iemocap/dataset/run/9/fold/9/eval/egemaps.pk

# 

# 
```



cat ./extract_egemaps.sh 

python extract_egemaps.py <dataset.wav.scp_path> <dataset.pk_path> <opensmile_dir_path>

python run_premodel.py <

python run_prognet.py

```



