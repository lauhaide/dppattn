""" Format candidate texts for FactCC evaluation """

import json
import os
import argparse
import spacy
nlp = spacy.load('en_core_web_sm')

def sentences(text):
    ret = []
    text_sentences = nlp(text)
    for sentence in text_sentences.sents:
        ret.append(sentence.text)
    return ret

def run(args):
    HOMEDIR = args.home

    outFile = 'data-dev.jsonl'

    modelOutputs = args.modelouts

    SRC = args.source

    srcTexts = open(os.path.join(HOMEDIR, SRC), 'r').readlines()

    for f in modelOutputs:
        modelOut = open(os.path.join(HOMEDIR, args.infolder, f), 'r')

        facts = []
        for e, (src, cand) in enumerate(zip(srcTexts, modelOut.readlines())):

            docs = src.strip().split('story_separator_special_tag')
            ssent = sentences(cand)
            for d, doc in enumerate(docs):
                for s, sent in enumerate(ssent):
                    factDict = {
                        'id': "ex{}-doc{}-s{}".format(e, d, s),
                        'text': doc,
                        'claim': sent,
                        'label': "CORRECT",
                    }
                    facts.append(factDict)


        newpath = os.path.join(HOMEDIR, args.outfolder, f.replace('.out', ''))
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        #dump dictionary
        with open(os.path.join(newpath, outFile), 'w') as outfile:
            for entry in facts:
                json.dump(entry, outfile)
                outfile.write('\n')
            outfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate FactCC input file.')
    parser.add_argument('--home', default='',  required = True,
                        help='Home directory where data is.')
    parser.add_argument('--source', default='test.txt.src',  required = True,
                        help='Which document should entail the content in the summary.')
    parser.add_argument('--infolder', default='testout',  required = True, help='Input files folder.')
    parser.add_argument('--outfolder', default='testout-factCC',  required = True, help='Output file folder.')

    #modelouts = ["val.newser-transformer-ori2-13000_no3B_nocovp_b1_debug.out",
    #                "val.newser-transformer-dpp-prevl-resca-14000_no3B_nocovp_b1_debug.out",
    #                "val.newser-transformer-covloss-14000_no3B_nocovp_b1_debug.out"]

    #modelouts = ["test.newser-brnn-ori-50000_4gpu_no3B.out",
    #          "test.newser-brnn-covloss-49000_no3B.out",
    #          "test.newser-brnn-covloss-vec-46000_no3B.out",
    #          "test.newser-brnn-dpp-49000_no3B.out",
    #          "test.newser-brnn-covloss-findpp-43000_no3B.out",
    #          "test.newser-transformer-ori2-14000_no3B.out",
    #          "test.newser-transformer-covloss-14000_no3B.out",
    #          "test.newser-transformer-dpp-prevl-resca-7000_no3B_temp06.out"]

    parser.add_argument('--modelouts', default=modelouts, help='Output file folder.')

    args = parser.parse_args()

    run(args)