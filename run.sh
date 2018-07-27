expdir="iemocap/tmp/exp/"

sed 's/expdir/$expdir/g' property.py \
  > $expdir/_property.py

cat import.py _property.py class_weight.py \
  > $expdir/_class_weight.py; chmod +x $expdir/_class_weight.py

exit # check

cat import.py $expdir/_property.py score.py \
  > _score.py

sed '/score.py/r $(./_score.py)' \
  > $expdir/_train.py

cat import.py property.py \
  > $expdir/_run.py; chmod +x _run.py

./_class_weight.py \
  >> _run.py; 

cat egemaps_dataset.py prognet.py $expdir/_train.py main.py \
  >> $expdir/_run.py


cat $expdir/_run.py \
  > $expdir/log

echo '"""' >> $expdir/log

$expdir/_run.py >> $expdir/log # run !

echo '"""' >> $expdir/log

rm $expdir/_class_weight.py
rm $expdir/_train.py
rm _score.py
