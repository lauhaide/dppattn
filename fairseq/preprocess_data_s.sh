#!/bin/bash

SRC_L=500
SRC_L_STR='L800'
CATEGORY=$1 #company, film, animal

TEXT=${HOME}'/'$CATEGORY'_tok_min5_L7.5k'

## single sequence source and target

DSTDIR=$CATEGORY'_tok_min5_L7.5k_tdtk_r2r_'$SRC_L_STR'_SS'
if [ ! -d "data-bin/$DSTDIR" ]; then
  cd data-bin/
  mkdir $DSTDIR
  cd ..
fi
python my_preprocess.py --source-lang src --target-lang tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/$DSTDIR \
  --nwordstgt 50000 --nwordssrc 50000 \
  --singleSeq --L $SRC_L \
  1> data-bin/$DSTDIR/preprocess.log

