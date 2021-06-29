# WikiCatSum
DATASET=$1
python preprocess.py -train_src ../../code/fairseq/data-bin/${DATASET}_tok_min5_L7.5k_tdtk_r2r_L800_SS_4onmt/train.src_bin_text.txt  \
                     -train_tgt ../../code/fairseq/data-bin/${DATASET}_tok_min5_L7.5k_tdtk_r2r_L800_SS_4onmt/train.tgt_bin_text.txt  \
                     -valid_src ../../code/fairseq/data-bin/${DATASET}_tok_min5_L7.5k_tdtk_r2r_L800_SS_4onmt/valid.src_bin_text.txt  \
                     -valid_tgt ../../code/fairseq/data-bin/${DATASET}_tok_min5_L7.5k_tdtk_r2r_L800_SS_4onmt/valid.tgt_bin_text.txt  \
                     -save_data ../../data/multinews/OpenNMT-Py-input/${DATASET}_r2r \
                     -src_seq_length 1000 \
                     -tgt_seq_length 1000 \
                     -src_seq_length_trunc 500 \
                     -tgt_seq_length_trunc 600 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 10000
