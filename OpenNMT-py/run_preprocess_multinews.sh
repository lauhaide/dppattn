#MultiNews
python preprocess.py -train_src ../../data/multinews/train.txt.src  \
                     -train_tgt ../../data/multinews/train.txt.tgt  \
                     -valid_src ../../data/multinews/val.txt.src    \
                     -valid_tgt ../../data/multinews/val.txt.tgt    \
                     -save_data ../../data/multinews/OpenNMT-Py-input/newser \
                     -src_seq_length 1000 \
                     -tgt_seq_length 1000 \
                     -src_seq_length_trunc 500 \
                     -tgt_seq_length_trunc 300 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 10000000