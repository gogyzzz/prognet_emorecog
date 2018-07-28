#!/bin/bash
if [ $# -ne 4 ]; then
  echo "Usage: $0 msp_improv/utt.list msp_improv/wav_cat.list msp_improv/ msp_improv/full_dataset.csv"
  exit 1;
fi

utt=$1
wavcat=$2
msp_dir=$3
csv=$4

spk="$msp_dir/spk.list"
sess="$msp_dir/sess.list"
gender="$msp_dir/gender.list"
emo="$msp_dir/cat.list"

gawk '{print $2}' $wavcat > $emo

cut -c6-8 $utt > $spk

cut -c6 $utt > $gender

cut -c2-3 $utt > $sess

# check
echo "<$emo>"; head -2 $emo
echo "<$spk>"; head -2 $spk
echo "<$gender>"; head -2 $gender
echo "<$sess>"; head -2 $sess

header="utterance,speaker,gender,session,emotion"

cat <(echo $header) <(paste -d ',' $utt $spk $gender $sess $emo) > $csv

echo "<$csv>"
head -3 $csv
