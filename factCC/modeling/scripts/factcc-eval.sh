#! /bin/bash
# Evaluate FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CODE_PATH=${HOME}/factCC/modeling # absolute path to modeling directory
export DATA_PATH=${HOME}/data/multinews/valout-factCC/val.newser-transformer-ori2-13000_no3B_nocovp_b1_debug  # absolute path to data directory
export SCORES_PATH=${HOME}/data/multinews/valout-scores # absolute path to data directory
export CKPT_PATH=${HOME}/factCC/factcc-checkpoint # absolute path to model checkpoint

export TASK_NAME=factcc_annotated
export MODEL_NAME=bert-base-uncased

echo ${DATA_PATH}

  python3 $CODE_PATH/run.py \
    --task_name $TASK_NAME \
    --do_eval \
    --eval_all_checkpoints \
    --do_lower_case \
    --overwrite_cache \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 12 \
    --model_type bert \
    --model_name_or_path $MODEL_NAME \
    --data_dir $DATA_PATH \
    --output_dir $CKPT_PATH 

