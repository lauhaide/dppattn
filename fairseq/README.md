
### Pre-process


After downloading the datasets you can generate binaries and dictionaries with the following command. You will need to define the variables as convenient.  

```TEXT``` should be the directory where to find the source and target texts  
```SRC_L``` is the length at which you will truncate the input sequence of paragraphs  

```
python my_preprocess.py --source-lang src --target-lang tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/$DSTDIR \
  --nwordstgt 50000 --nwordssrc 50000 \
  --singleSeq --L $SRC_L \
  1> data-bin/$DSTDIR/preprocess.log
```

Note: need to add ```--out-idx-train``` and ```--out-idx-valid``` when preparing WikiCatSum for OpenNMT -based models.


### Train


To train the base CVS2S model:

```
CUDA_VISIBLE_DEVICES=$GPUID python train.py data-bin/$DATADIR --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_iwslt_de_en --save-dir checkpoints/$MODELNAME  --skip-invalid-size-inputs-valid-test --no-progress-bar --task translation --max-target-positions $MAX_TGT_SENT_LEN --max-source-positions MAX_SRC_POSITIONS --outindices checkpoints/$IDXEXCLDIR/ignoredIndices.log --outindicesValid $OUTDIR$IDXEXCLDIR/valid_ignoredIndices.log 1> 'checkpoints/'$MODELNAME'/train.log'
```

```--outindices``` and ```--outindicesValid``` should point to files with list of excluded instances' indices. You should define the other variables as convenient.


For the different variants use the following arguments:

- (+CovLoss) add ```--criterion cross_entropy_covloss```  
- (+DPP) change ```--arch``` to ```fconvdpp_prevl_iwslt_de_en ```  



### Generate

```
CUDA_VISIBLE_DEVICES=2 python my_generateSingle.py data-bin/$DATADIR --path checkpoints/$MODELNAME/checkpoint_best.pt --beam 5 --skip-invalid-size-inputs-valid-test --decode-dir $DECODEDIR --reference-dir $REFDIR --outindices $IDXEXCLDIR/valid_ignoredIndices.log --max-target-positions $MAX_TGT_SENT_LEN --quiet --gen-subset valid --ngram 3 1> $DECODEDIR/generate.log
```

For greedy decoding, set ```--beam 1``` and remove ```--ngram 3```.

You can also select best checkpoint based on ROUGE on valid:
```
export ARG_LIST="--beam 5 --skip-invalid-size-inputs-valid-test --reference-dir $REFDIR --outindices $IDXEXCLDIR/valid_ignoredIndices.log --max-target-positions $MAX_TGT_SENT_LEN --quiet "

CUDA_VISIBLE_DEVICES=$GPUID python run_dev_rouge.py \
--data-dir data-bin/$DATADIR \
--model-dir checkpoints/$MODELNAME \
--reference-dir $REFDIR \
--fconv
```

