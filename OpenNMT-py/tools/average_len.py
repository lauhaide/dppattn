import json
import os
import argparse
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')

def getSentences(text):
    ret = []
    text_sentences = nlp(text)
    for sentence in text_sentences.sents:
        ret.append(sentence.text)
    return ret

def run(args):

    for f in args.modelouts:
        print(f)
        modelOut = open(os.path.join(args.home, args.infolder, f), 'r')

        tokens = []
        sentences = []
        for e, cand in enumerate(modelOut.readlines()):
            cand = cand.strip()
            ssent = getSentences(cand)
            sentences.append(len(ssent))
            tokens.append(len(cand.split()))

        modelStats = open(os.path.join(args.home, args.outfolder, f.replace('.out', '.stats')), 'w')

        npScores = np.array(sentences)
        modelStats.write("Nb. Sentences Mean {}\nMin {}\nMax {}\nStd {}\nLen:{}\n\n".format(npScores.mean(), \
                                                               npScores.min(), \
                                                               npScores.max(), npScores.std(), len(sentences)))
        npScores = np.array(tokens)
        modelStats.write("Nb. Tokens Mean {}\nMin {}\nMax {}\nStd {}\nLen:{}\n\n".format(npScores.mean(), \
                                                               npScores.min(), \
                                                               npScores.max(), npScores.std(), len(tokens)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output average length.')
    parser.add_argument('--home', default='', required = True,
                        help='Home directory where data is.')
    parser.add_argument('--infolder', default='testout', required = True, help='Input files folder.')
    parser.add_argument('--outfolder', default='testout-scores', required = True, help='Output file folder.')

    # modelouts = ["test.newser-brnn-ori-50000_4gpu_no3B.out",
    #          "test.newser-brnn-covloss-49000_no3B.out",
    #          "test.newser-brnn-covloss-vec-46000_no3B.out",
    #          "test.newser-brnn-dpp-49000_no3B.out",
    #          "test.newser-brnn-covloss-findpp-43000_no3B.out",
    #          "test.newser-transformer-ori2-14000_no3B.out",
    #          "test.newser-transformer-covloss-14000_no3B.out",
    #          "test.newser-transformer-dpp-prevl-resca-7000_no3B_temp06.out"]

    # modelouts = ["val.newser-transformer-ori2-13000_no3B_nocovp_b1_debug.out",
    #                "val.newser-transformer-dpp-prevl-resca-14000_no3B_nocovp_b1_debug.out",
    #                "val.newser-transformer-covloss-14000_no3B_nocovp_b1_debug.out"]

    parser.add_argument('--modelouts', default=modelouts, help='Output file folder.')

    args = parser.parse_args()

    run(args)