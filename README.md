Code and Data for [Multi-Document Summarization with Determinantal Point Process Attention]().


## Datasets

We use the *WikiCatSum* dataset available [here](https://datashare.is.ed.ac.uk/handle/10283/3368). In particular, for our controlled experiments we use an Oracle (Section 4 in the paper) to rank the input and then truncate it to 500 input tokens. 

We use the *MultiNews* data as preprocessed by Fabbri et al. (2019) ([here](https://github.com/Alex-Fabbri/Multi-News)).


## Code and Model Training

Our code extends implementations in OpenNMT (Pointer-Generator and Transformer) [here](OpenNMT-py/) and Fairseq (ConvSeq2Seq) [here](fairseq/) to use DPP attention.


### Evaluation

#### ROUGE

We use the wrapper script [test_rouge.py](OpenNMT-py/tools/test_rouge.py) as used in [MultiNews](https://github.com/Alex-Fabbri/Multi-News).

#### BERTScore

We installed [BERTScore](https://arxiv.org/abs/1904.09675) with ```pip install bert-score``` (version 0.3.9). Our script to run BERTScore: [run_bertscore.sh](OpenNMT-py/tools/run_bertscore.sh) (and previous formatting needed by Fariseq outputs is done by running this Python script: [format-fairseqout-to-bertscore](OpenNMT-py/tools/format-fairseqout-to-bertscore.py)).

#### Sentence Movers' Similarity

For the [sentence mover's similarity metrics](https://www.aclweb.org/anthology/P19-1264) we follow the code here [SMS github](https://github.com/eaclark07/sms). Our script to run this metric: [run_sms.sh](OpenNMT-py/tools/run_sms.sh).

#### Fact_acc MultiNews

We adapt the model proposed in [Neural Text Summarization: A Critical Evaluation](https://www.aclweb.org/anthology/D19-1051/) for multi-document evaluation. Installation instructions and the trained model can be found in [FACTCC github](https://github.com/salesforce/factCC). You will need to run [format-to-factCC-eval.py](OpenNMT-py/tools/format-to-factCC-eval.py) to format model outputs as expected, **factcc-eval.sh** (with updated directory references from factCC/modeling/scripts/) to run the model evaluation, and [factCC-summarise-predictions.py](OpenNMT-py/tools/factCC-summarise-predictions.py) to summarise results. Note that we provide our modified version of FactCC [run.py](factCC/modeling/run.py).


#### Fact_acc WikiCatSum

We implement the Fact_acc metric from [Assessing the factual accuracy of generated text](https://dl.acm.org/doi/10.1145/3292500.3330955) and use the relation extraction system proposed by [(Sorokin and Gurevych, 2017)](https://www.aclweb.org/anthology/D17-1188.pdf) available at [Relation Extraction github](https://github.com/UKPLab/emnlp2017-relation-extraction). For installation follow instructions provided there. Our script to run this metric is [run_relext.sh](OpenNMT-py/tools/run_relext.sh).


## Outputs

[Models Outputs](https://drive.google.com/file/d/1jjyUtEFvvDBL6ni0dDcqoDVf3NRqsqai/view?usp=sharing)

