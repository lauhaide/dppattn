#
# This should be copied to the folder where the system (https://github.com/UKPLab/emnlp2017-relation-extraction) was cloned.
#

import numpy as np
import os
import argparse
import sys
sys.path.insert(0, "relation_extraction/")


from pycorenlp import StanfordCoreNLP
from relation_extraction.core import entity_extraction
from relation_extraction.core.parser import RelParser
from relation_extraction.core import keras_models


corenlp = StanfordCoreNLP('http://localhost:9000/')
corenlp_properties = {
    'annotators': 'tokenize, pos, ner',
    'outputFormat': 'json'
}

# does not take the following, so modified its source files to harcode this embeddings:
keras_models.model_params['wordembeddings'] = "/home/lperez/wikigen/code/emnlp2017-relation-extraction/resources/embeddings/glove/glove.6B.50d.txt"
relparser = RelParser("model_ContextWeighted", models_folder="trainedmodels/")

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('/home/lperez/storage/pretrained/mbart.trimXWikis2/sentence.bpe.model')

def get_tagged_from_server(input_text):
    """
    Send the input_text to the CoreNLP server and retrieve the tokens, named entity tags and part-of-speech tags.
    """
    annotation = corenlp.annotate(input_text,properties=corenlp_properties).get("sentences", [])
    tagged = []
    for i in range(len(annotation)):
        corenlp_output = annotation[i]
        tagged.append([(t['originalText'], t['ner'], t['pos']) for t in corenlp_output['tokens']])
    return tagged

def score(graph_ref, graph_cand, f):
    # {'tokens': ['Germany', 'is', 'a', 'country', 'in', 'Europe'], 'edgeSet':
    # [{'left': [0], 'right': [5], 'kbID': 'P30', 'lexicalInput': 'continent'},
    # {'left': [0], 'right': [3], 'kbID': 'P0', 'lexicalInput': 'ALL_ZERO'},
    # {'left': [5], 'right': [3], 'kbID': 'P31', 'lexicalInput': 'instance of'}]}

    def getTripleSet(graph):
        triples = set()
        keys = set()
        relLabels = set()
        for g in range(len(graph)):
            for edge in graph[g]['edgeSet']:
                str = [graph[g]['tokens'][i] for i in edge['left']]
                str_key = "-".join(str) + '#' + edge['kbID']
                str_val = " ".join([graph[g]['tokens'][i] for i in edge['right']])
                triples.add( (str_key, str_val) )
                keys.add(str_key)
                relLabels.add(edge['kbID'] + '#' + edge['lexicalInput'])
        return triples, keys, relLabels

    triples_ref, keys, labels_ref = getTripleSet(graph_ref)
    triples_cand, _, labels_cand = getTripleSet(graph_cand)

    Ft_Fg = 0
    Fg = 0
    for key, value in triples_cand:
        if key in keys:
            for k, v in triples_ref:
                if key==k and (value in v or v in value):
                    Ft_Fg += 1
            Fg += 1

    f.write('{' + ', '.join(["('{}', '{}')".format(x[0], x[1]) for x in triples_ref ]) + '}\n')
    f.write('{' + ', '.join(["('{}', '{}')".format(x[0], x[1]) for x in triples_cand ]) + '}\n')

    factcc = 0 if Fg==0 else Ft_Fg / Fg
    f.write('Ft_Fg/Fg={}   Fg={}\n'.format(factcc, Fg))

    return factcc, Fg, (labels_ref, labels_cand)

def main(args):

    def getParsedGraph(tagged):
        parsed_graphs = []
        for i in range(len(tagged)):
            entity_fragments = entity_extraction.extract_entities(tagged[i])
            edges = entity_extraction.generate_edges(entity_fragments)
            non_parsed_graph = {'tokens': [t for t, _, _ in tagged[i]],
                                'edgeSet': edges}
            parsed_graph = relparser.classify_graph_relations([non_parsed_graph])
            if parsed_graph:
                parsed_graphs.append(parsed_graph[0])
        return parsed_graphs



    fref = open(args.ref, 'r')
    ref_graphs = []
    for e, ref in enumerate(fref.readlines()):
        tagged = get_tagged_from_server(sp.decode_pieces(sp.encode_as_pieces(ref.strip())))
        parsed_graph_ref = getParsedGraph(tagged)
        ref_graphs.append(parsed_graph_ref)
        #if e>10:
        #    break

    for candfile in args.cand:
        print(candfile)

        fcand = open(candfile, 'r')
        if args.system == 'opennmt':
            modelout = candfile.split('/')[-1].split('.out')[0]
            print(os.path.join(args.outdir, modelout + '.relext2'))
            outcand = open(os.path.join(args.outdir, modelout + '.relext2'), 'w')
        else:
            modelout = candfile.split('/')[-2]
            print(os.path.join(args.outdir, modelout + '.relext'))
            outcand = open(os.path.join(args.outdir, modelout + '.relext'), 'w')
        ignored = []
        scores = []
        FGs = []
        labels_ref = set()
        labels_cand = set()
        for ex, (parsed_graph_ref, cand) in enumerate(zip(ref_graphs, fcand.readlines())):

            tagged = get_tagged_from_server(sp.decode_pieces(sp.encode_as_pieces(cand.strip())))
            parsed_graph_cand = getParsedGraph(tagged)

            if parsed_graph_ref and parsed_graph_cand:
                factcc, Fg, (lr, lc) = score(parsed_graph_ref, parsed_graph_cand, outcand)
                scores.append(factcc)
                FGs.append(Fg)
                labels_ref.update(lr)
                labels_cand.update(lc)
            elif not parsed_graph_ref:
                outcand.write('ref {} {}\n'.format(ex, ref))
                ignored.append(ex)
            else:
                outcand.write('Counting 0 for cand: {} \n{}\n'.format(ex, cand))
                #ignored.append(ex)
                scores.append(0)
                FGs.append(0)

        npScores = np.array(scores)
        outcand.write("Fact_cc Mean {}\nMin {}\nMax {}\nStd {}\nLen:{}\n".format(npScores.mean(), \
                                                               npScores.min(), \
                                                               npScores.max(), npScores.std(), len(scores)))
        npScores = np.array(FGs)
        outcand.write("FG Mean {}\nMin {}\nMax {}\nStd {}\nLen:{}\n".format(npScores.mean(), \
                                                               npScores.min(), \
                                                               npScores.max(), npScores.std(), len(FGs)))
        outcand.write('Ignored {}\n'.format(len(ignored)))
        outcand.write(' '.join([str(x) for x in ignored]) + '\n')

        outcand.write("Relation labels Ref: [{}]\n".format(", ".join(["'{}'".format(x) for x in labels_ref])))
        outcand.write("Relation labels Cand: [{}]\n".format(", ".join(["'{}'".format(x) for x in labels_cand])))
        outcand.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ref', help="reference file", required=True)
    parser.add_argument('--cand', help="candidate files", required=True, action='append')
    parser.add_argument('--outdir', help="output folder", required=True)
    parser.add_argument('--system', help="candidate files from Fairseq or OpenNMT", required=True)


    args = parser.parse_args(sys.argv[1:])
    main(args)
