
### Pre-process

You can pre-process the datasets after downloading them with [run_preprocess_multinews.sh](run_preprocess_multinews.sh) and [run_preprocess_wikicatsum.sh](run_preprocess_wikicatsum.sh) to generate binaries and dictionaries. Note that WikiCatSum needs some previous pre-processing, you will need to look at the script [here](../fairseq/preprocess_data_s_onmt.sh). You will need to define the variables as convenient.  



### Train

#### Pointer-Generator models

The following is the command to train the base Pointer-Generator (PG) model variant on the Animal dataset:
```
MODEL='animal_r2r-brnn-ori'
DATASET='animal_r2r'

python train.py -save_model ${HOME}/models/${MODEL}/${DATASET} \
    -data ${HOME}/OpenNMT-Py-input/${DATASET} \
    -encoder_type brnn -word_vec_size 128 -rnn_size 512 -layers 1 \
    -copy_attn -global_attention mlp \
    -train_steps 50000 -max_grad_norm 4 -dropout 0.  -optim adagrad \
    -learning_rate 0.15 -adagrad_accumulator_init 0.1 \
    -accum_count 8 -batch_size 5 \
    -reuse_copy_attn -copy_loss_by_seqlength \
    -bridge -seed 777 -gpu_ranks 0 1  -world_size 2 \
    -valid_batch_size 8 -save_checkpoint_steps 1000 -valid_steps 1000 -early_stopping 5 -keep_checkpoint 20 \
    -lambda_coverage 0 \
    -log_file ${HOME}/logs/${MODEL}.log -log_file_level 'INFO'
```

For the different variants replace line ```-lambda_coverage 0``` with one of the followings:

- (+CovLoss) ```-coverage_attn -lambda_coverage 1```  
- (+CovLossVec) ```-coverage_attn -lambda_coverage 1 -coverage_attn_feed```  
- (+DPP) ```-lambda_coverage 0 -dpp_attention -dpp_rescaled ```

This command is also used for the other datasets (updating ```DATASET```) accordingly for Company or Film. For Film, we use ```-world_size 3``` and save-checkpoint and valid steps set to 500.


#### Copy Transformer


The following is the command to train the base Pointer-Generator (PG) model variant on the Animal dataset:
```
MODEL='animal_r2r-transformer-ori'
DATASET='animal_r2r'

python train.py \
-save_model ${HOME}/models/${MODEL}/${DATASET} \
-data ${HOME}/OpenNMT-Py-input/${DATASET} \
-copy_attn -word_vec_size 512 -rnn_size 512 -layers 4 -encoder_type transformer -decoder_type transformer \
-position_encoding -train_steps 50000 -warmup_steps 6000 -learning_rate 2 -decay_method noam -label_smoothing 0.1 \
-max_grad_norm 0 -dropout 0\.2 -batch_size 3072 -optim adam -adam_beta2 0.998 -param_init 0 -batch_type tokens \
-normalization tokens -max_generator_batches 2 -accum_count 4 -share_embeddings -param_init_glorot \
-seed 777 -gpu_ranks 0 1  -world_size 2 \
-valid_batch_size 10 -valid_steps 1000 -save_checkpoint_steps 1000 -early_stopping 5 -keep_checkpoint 10 \
-lambda_coverage 0 \
-log_file logs/${MODEL}.log -log_file_level 'INFO'

```

For the different variants use the following arguments:

- (+CovLoss) ```-coverage_attn -lambda_coverage 1```  
- (+DPP) ```-lambda_coverage 0 -dpp_rescaled -decoder_type transformerDPPPrevl ``` (Animal, MultiNews)  
- (+DPP) ```-lambda_coverage 0 -dpp_rescaled -decoder_type transformerDPPPrevlP ``` (Film, Company)




### Generate

Below the generation commands we used for all OpenNMT models and datasets.

Decoding with beam=5, trigram-block and length and coverage penalties.

```
MODEL='animal_r2r-brnn-covloss'
DATASET=animal
STEP=44000
MAXDECLEN=200

python translate.py -gpu 0 -batch_size 50 -beam_size 5 \
   -model ${HOME}/models/${MODEL}/${DATASET}_r2r_step_${STEP}.pt \
   -src ${HOME}/data-bin/${DATASET}_tok_min5_L7.5k_tdtk_r2r_L800_SS_4onmt/test.src_bin_text.txt \
   -output ${HOME}/testout/test.${MODEL}-${STEP}'_nogehr.out' \
   -max_length ${MAXDECLEN} -length_penalty wu -alpha 0.9 \
   -block_ngram_repeat 3 -ignore_when_blocking "." "</t>" "<t>"
```

Note: 
<ol>
<li> Film and Company no ```-length_penalty``` is used (alpha=0). </li>
<li> For Multinews, no ```-block_ngram_repeat``` is used, instead ```-stepwise_penalty -coverage_penalty summary -beta 5``` is used.</li>
<li> WikiCatSum: MAXDECLEN=200 for brnn models, MAXDECLEN=600 for transformer models.</li>
<li> MultiNews: MAXDECLEN=300 and ```-min_length 200``` </li>
<li> ```-dectemp 0.6``` for CTF+DPP. </li>
</ol>



The following is an example command for greedy decoding, no repetition constraints.

```
DATASET=animal
MODEL='animal_r2r-brnn-covloss'
STEP=34000
MAXDECLEN=200


python translate.py -gpu 0 -batch_size 50 -beam_size 1 \
-model ${HOME}/models/${MODEL}/${DATASET}_r2r_step_${STEP}.pt \
-src ${HOME}/data-bin/${DATASET}_tok_min5_L7.5k_tdtk_r2r_L800_SS_4onmt/valid.src_bin_text.txt \
-output ${HOME}/valout/b1/val.${MODEL}-${STEP}'_b1.out' \
-max_length ${MAXDECLEN}
```


