///////////////////////////////////////////////////////////////////////////////////////
///////// > openSMILE configuration file for data output <           //////////////////
/////////                                                            //////////////////
///////// (c) 2014-2016 audEERING,                                   //////////////////
/////////     All rights reserverd. See file COPYING for details     //////////////////
///////////////////////////////////////////////////////////////////////////////////////


/*
   This file can be included as data output file for standard feature
   extraction configuration files. It provides commandline options
   for the batch extraction GUI, and supports LLD and Functionals (summaries)
   saving.

   It requires the main extrator configuration file to provide the following
   data memory levels:  lld, lld_de, and func
 */

;;;;;;;;; output LLD features to CSV ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

[componentInstances:cComponentManager]
instance[htksink].type=cHtkSink

  ; TODO: output without a frame period does not work so that HTK (HList) can read it!
[htksink:cHtkSink]
reader.dmLevel = func
filename=\cm[htkoutput(O){output.sink.htk}:output HTK file for summarised parameters, set to a valid filename to enable this output sink, no append by default, use -appendstatichtk option to enable]
append=\cm[appendhtk{0}:set to 1 to enable appending to an existing HTK parameter summary file, given by the htkoutput option]
parmKind = 9
  ; this is broken for HTK sink...
errorOnNoOutput = 0
 ; avoid broken files which HTK cannot read for period approx. > 0.06 (int16 overflow)
forcePeriod = 0.01

