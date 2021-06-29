#
# This should be copied to the folder where the system (https://github.com/UKPLab/emnlp2017-relation-extraction) was cloned.
#

source activate relext #APPROPRIRATE ENVIRONMENT!



HOMEDIR='/YOUR HOME TO THIS DATA/data/multinews'
SCORES_FOLDER='testout-scores'
CANDI_FOLDER='testout'
system='opennmt'  #'fairseq'  'opennmt'

#WCS4ONMT='/YOUR HOME TO THIS DATA/fairseq/data-bin'

#Animal
#REFFILE=${WCS4ONMT}'/animal_tok_min5_L7.5k_tdtk_r2r_L800_SS_4onmt/test.tgt_bin_text.txt'
#declare -a PREFIX_LIST_SRC=("test.animal_r2r-transformer-ori-10000_nogehr" \
#"test.animal_r2r-transformer-covloss-10000_nogehr" "test.animal_r2r-transformer-dpp-prevl-resca-13000_nogehr_temp06" \
#"test.animal_r2r-brnn-dpp-resca-49000_nogehr" "test.animal_r2r-brnn-ori-48000_nogehr" \
#"test.animal_r2r-brnn-covloss-44000_nogehr" "test.animal_r2r-brnn-covloss-vec-49000_nogehr")


#Company
#REFFILE=${WCS4ONMT}'/company_tok_min5_L7.5k_tdtk_r2r_L800_SS_4onmt/test.tgt_bin_text.txt'
#declare -a PREFIX_LIST_SRC=("test.company_r2r-transformer-ori-11000_nogehr" \
#"test.company_r2r-transformer-covloss-11000_nogehr" \
#"test.company_r2r-transformer-dpp-prevl-past-resca-11000_nogehr_temp06" "test.company_r2r-brnn-ori-40000_nogehr" \
#"test.company_r2r-brnn-covloss-38000_nogehr" "test.company_r2r-brnn-covloss-vec-36000_nogehr" \
#"test.company_r2r-brnn-dpp-resca-49000_nogehr")

#Film
#REFFILE=${WCS4ONMT}'/film_tok_min5_L7.5k_tdtk_r2r_L800_SS_4onmt/test.tgt_bin_text.txt'
#declare -a PREFIX_LIST_SRC=( "test.film_r2r-transformer-ori-2gpu-7000_nogehr" \
#"test.film_r2r-transformer-covloss-2gpu-7000_nogehr" \
#"test.film_r2r-transformer-dpp-prevl-past-resca-6500_nogehr_temp06" \
#"test.film_r2r-brnn-ori-43000_nogehr" "test.film_r2r-brnn-covloss-35000_nogehr" \
#"test.film_r2r-brnn-covloss-vec-47000_nogehr" "test.film_r2r-brnn-dpp-resca-onmtpy-32500_nogehr")


# On Fairseq ConS2S ouputs
#FAIRHOMEDIR='/YOUR HOME TO THIS DATA/fairseq/outputs'
#FAIRHOMEDIRTEST=${FAIRHOMEDIR}'/test'

# Animal
#REFFILE=${FAIRHOMEDIR}'/reference/animal_test_all.ref'
#declare -a PREFIX_LIST_SRC=("fconvdpp_prevl_iwslt_de_en_tdtk_r2r_animal_L800_checkpoint18" \
#"fconv_iwslt_de_en_covloss__tdtk_r2r_animal_L800_checkpoint16" \
#"fconv_iwslt_de_en_tdtk_r2r_animal_L800_checkpoint15")

# Company
#REFFILE=${FAIRHOMEDIR}'/reference/company_test_all.ref'
#declare -a PREFIX_LIST_SRC=("fconvdpp_prevl_iwslt_de_en_tdtk_r2r_company_L800_checkpoint13" \
#"fconv_iwslt_de_en_covloss__tdtk_r2r_company_L800_checkpoint11" \
#"fconv_iwslt_de_en_tdtk_r2r_company_L800_checkpoint14")

# Film
#REFFILE=${FAIRHOMEDIR}'/reference/film_test_all.ref'
#declare -a PREFIX_LIST_SRC=("fconvdpp_prevl_iwslt_de_en_tdtk_r2r_film_L800_checkpoint14" \
#"fconv_iwslt_de_en_covloss__tdtk_r2r_film_L800_checkpoint14" \
#"fconv_iwslt_de_en_tdtk_r2r_film_L800_checkpoint14")


for CANDI in ${PREFIX_LIST_SRC[@]}; do

    if [ $system == 'opennmt' ]; then
       CANDIS+=' --cand '${HOMEDIR}/${CANDI_FOLDER}/${CANDI}.out
    else
      CANDIS+=' --cand '${FAIRHOMEDIRTEST}/${CANDI}/all.dec
    fi

done


python relext-factacc.py --ref ${REFFILE} \
      ${CANDIS} \
      --outdir ${HOMEDIR}/${SCORES_FOLDER} \
      --system $system
