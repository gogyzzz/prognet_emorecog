#!/bin/bash

osmdir=$1
wavcat=$2
egemapshtk=$3

conf=$osmdir/config/gemaps/eGeMAPSv01a.conf

wavhtk=wav_htk.list.tmp
wav=wav.list.tmp

awk '{print $1}' $wavcat > $wav
echo ""
echo "<$wav>"
head -2 $wav

paste -d ' ' $wav $egemapshtk > $wavhtk
echo ""
echo "<$wavhtk>"
head -2 $wavhtk

cat $wavhtk | parallel --colsep ' ' $osmdir/SMILExtract -C "$conf" -I {1} -O {2}

rm $wav
rm $wavhtk
