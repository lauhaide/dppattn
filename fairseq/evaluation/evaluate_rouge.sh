export ROUGE=${HOME}/path/to/pyrouge/tools/ROUGE-1.5.5/
export REFDIR=${HOME}/fairseq/outputs/reference/film_valid
python evaluation/eval_full_model.py --rouge --cmd_rouge '-c 95 -r 1000 -n 4 -m -U -u -2 4' \
--decode_dir ${HOME}/fairseq/outputs/fconvdpp_prevl_iwslt_de_en_tdtk_r2r_film_L800_checkpoint14_ngram


