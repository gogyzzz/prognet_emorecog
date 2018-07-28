#!/bin/bash

osmdir=$1
myconfdir="opensmile_configs"
cp $myconfdir/standard_wave_input.conf.inc $osmdir/config/shared/.
cp $myconfdir/wav_to_88egemaps_htk.conf.inc $osmdir/config/shared/.
cp $myconfdir/eGeMAPSv01a.conf $osmdir/config/gemaps/.
