# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import numpy as np
from operator import itemgetter

import fairseq.tokenizer as tokenizer
import fairseq.data.data_utils as du
from fairseq.preprocess.Constants import EOP, SNT, BLANK, EOT
from fairseq.tokenizerWCSSentence import TokenizerWCSSentence

from evaluation.rouge import rouge_n_recall


import random

class TokenizerWCSSingleSequence(tokenizer.Tokenizer):

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenizer.tokenize_line,
                 append_eos=True, reverse_order=False, L=None,
                 aconsumer=None, annotator=None, outfileTexts=False, excludIndex=[]):
        """
        :param filename:
        :param dict:
        :param consumer:
        :param tokenize:
        :param append_eos:
        :param reverse_order:
        :param L: maximum length of the input in tokens
        :return:
        """


        ntok = 0
        replaced = Counter()
        outt = None

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        print("*    Truncating to L={}".format(L))
        with open(filename, 'r') as f:
            if outfileTexts:
                print("*    Write out the binarised texts")
                outt = open(filename+"_bin_text.txt", 'w')
            ex = 0
            huevo=0
            for line in f:

                # indices of instances to be excluded
                if ex in excludIndex:
                    ex +=1
                    continue

                # As used in WikiSum inputs: "Construct inputs from Wiki title and references"
                # title tokens go at the begining, just remove separation token
                # source lines might have "<EOT>" separating the title at the beginning
                # leave it as WikiSum does

                line = line.replace(EOP, BLANK) # paragraphs markers are removed in hierarchical input format, also in sigle seqs
                line = line.replace(SNT, BLANK) # sentence markers are neither in single seqs, this is for tgt

                # take the fist L tokens from the source texts
                if L:
                    tokLine = line.split()
                    if len(tokLine) > L:
                        tokLine = tokLine[:L]
                        line = " ".join(tokLine)

                # log binarised strings
                if outt is not None:
                    outt.write(line.strip() + "\n")

                ids = tokenizer.Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )

                consumer(ids)
                ntok += len(ids)
                ex +=1
            if outfileTexts:
                outt.close()

        return {'ntok': ntok,  'nunk': sum(replaced.values()), 'replaced': len(replaced),
                '#examples':ex,
                }

    @staticmethod
    def chunks_r2(phrase, n, tgt):
        """Yield successive n-sized chunks from l."""
        l = phrase.split()
        return [(" ".join(l[i:i + n]), rouge_n_recall([l[i:i + n]], [tgt]) ) for i in range(0, len(l), n)]



    @staticmethod
    def binarizeRepeat(filename, dict, consumer, tokenize=tokenizer.tokenize_line,
                 append_eos=True, reverse_order=False, L=None,
                 aconsumer=None, annotator=None, R2Rrerank=False, reptype=None,
                 outfileTexts=False, excludIndex=[]):
        """

        Generates synthetic data with different levels of repetition:

        - A) token repetition (low)
        - B) chunk repetition (medium literal) 2 chunks
        - C) chunk repetition (high literal) 7 chunks
        - D) paragraph repetition (high related, choose paragraph with greater ROUGE-L overlap)

        Repetitions are added to a base of best N input paragraphs (up to L=800 tokens)

        :param filename:
        :param dict:
        :param consumer:
        :param tokenize:
        :param append_eos:
        :param reverse_order:
        :param L: maximum length of the input in tokens
        :return:
        """

        random.seed(30)
        ntok = 0
        replaced = Counter()
        r2rScores = None
        outt = None

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        print("*    synthetic data type: {}".format(reptype))

        with open(filename, 'r') as f:
            if outfileTexts:
                print("*    Write out the binarised texts")
                outt = open(filename+"_bin_text.txt", 'w')

            if L and R2Rrerank:
                r2rScoresFile =  open(filename.replace('src','r2r.src'), 'r')
                r2rScores = r2rScoresFile.readlines()
                if reptype == "D":
                    tgtsFile = open(filename.replace('src','tgt'), 'r')
                    tgts = tgtsFile.readlines()

            ex = 0
            for line in f:

                # indices of instances to be excluded
                if ex in excludIndex:
                    ex +=1
                    continue

                # As used in WikiSum inputs: "Construct inputs from Wiki title and references"
                # title tokens go at the begining, just remove separation token
                # source lines might have "<EOT>" separating the title at the beginning
                # leave it as WikiSum does

                if r2rScores:
                    plines = line.split(EOT)
                    title = plines[0]
                    plines = plines[1].split(EOP)
                    pscores = r2rScores[ex].split(EOP)
                    assert len(pscores)==len(plines), "scores={}; lines={}".format(len(pscores), len(plines))
                    sortedIndex = np.argsort(pscores)[::-1]
                    plines = [plines[i].replace("\n", " ") for i in sortedIndex]
                    line = title + EOT + " ".join(plines)
                    if reptype == "D":
                        tgt = tgts[ex].replace(SNT,BLANK).split()
                else:
                    # paragraphs markers are removed in hierarchical input format, also in sigle seqs
                    line = line.replace(EOP, BLANK)

                line = line.replace(SNT, BLANK) # sentence markers are neither in single seqs, this is for tgt

                # take the fist L tokens from the source texts
                if L:
                    tokLine = line.split()
                    if len(tokLine) > L:
                        tokLine = tokLine[:L]
                        if reptype == 'A':
                            print("not yet!")
                            exit()

                        elif reptype == 'B':
                            print("B")
                            exit()

                            line = " ".join(tokLine)
                            xphrases = TokenizerWCSSentence.chunks(line, 40)
                            all_phrases = list(range(len(xphrases)))
                            replace_chunks = random.sample(all_phrases, 2)
                            rest_all_phrases = [x for x in all_phrases if x not in replace_chunks]
                            random.shuffle(rest_all_phrases)

                            chunk1 = xphrases[replace_chunks[0]]
                            chunk2 =  xphrases[replace_chunks[1]]

                            xphrases.insert(rest_all_phrases[0], chunk1)
                            xphrases.insert(rest_all_phrases[1], chunk2)

                            line = " ".join(xphrases)

                        elif reptype == 'B1':
                            print("**B1**")
                            exit()

                            line = " ".join(tokLine)
                            xphrases = TokenizerWCSSentence.chunks(line, 40)
                            all_phrases = list(range(len(xphrases)))
                            replace_chunks = random.sample(all_phrases, 2)
                            rest_all_phrases = [x for x in all_phrases if x not in replace_chunks]
                            random.shuffle(rest_all_phrases)

                            chunk1 = xphrases[replace_chunks[0]]
                            chunk2 =  xphrases[replace_chunks[1]]

                            chunks = [chunk1]*3 + [chunk2]*3
                            random.shuffle(chunks)

                            if len(rest_all_phrases) > 6:
                                random.shuffle(rest_all_phrases)
                                for g in range(6): # insert in first 6 random positions
                                    xphrases.insert(rest_all_phrases[g], chunks[g])
                            else:
                                # if remainin chunks are less than 6, insert in between the existing positions and
                                # add the reminder at the end
                                for g in range(len(rest_all_phrases)):
                                    xphrases.insert(rest_all_phrases[g], chunks[g])
                                for g in range(len(rest_all_phrases),6):
                                    xphrases.append(chunks[g])

                            line = " ".join(xphrases)

                        elif reptype == 'C':
                            print("C")
                            exit()

                            line = " ".join(tokLine)
                            xphrases = TokenizerWCSSentence.chunks(line, 40)
                            all_phrases = list(range(len(xphrases)))
                            replace_chunks = random.sample(all_phrases, min(7, len(xphrases)))
                            rest_all_phrases = [x for x in all_phrases if x not in replace_chunks]

                            chunks = []
                            for g in replace_chunks:
                                chunks.append(xphrases[g])

                            if len(rest_all_phrases) > 7:
                                random.shuffle(rest_all_phrases)
                                for g in range(7): # insert in first 7 random positions
                                    xphrases.insert(rest_all_phrases[g], chunks[g])
                                    print("\n",chunks[g])
                            else:
                                # if remainin chunks are less than 7, insert in between the existing positions and
                                # add the reminder at the end
                                for g in range(len(rest_all_phrases)):
                                    xphrases.insert(rest_all_phrases[g], chunks[g])
                                for g in range(len(rest_all_phrases),7):
                                    xphrases.append(chunks[g])

                            line = " ".join(xphrases)

                        elif reptype == 'D':
                            print("D")
                            exit()

                            line = " ".join(tokLine)
                            xphrases_tmp = TokenizerWCSSingleSequence.chunks_r2(line, 40, tgt)
                            xphrases = [x[0] for x in xphrases_tmp]
                            xphrases_tmp = sorted(xphrases_tmp, key=itemgetter(1), reverse=True)

                            chunk1 = xphrases_tmp[0]
                            chunk2 = xphrases_tmp[1]

                            chunks = [chunk1[0]]*3 + [chunk2[0]]*3
                            random.shuffle(chunks)

                            rest_all_phrases = list(range(len(xphrases)))
                            random.shuffle(rest_all_phrases)

                            #xphrases.insert(rest_all_phrases[0], chunk1[0])
                            #xphrases.insert(rest_all_phrases[1], chunk2[0])

                            if len(rest_all_phrases) > 6:
                                random.shuffle(rest_all_phrases)
                                for g in range(6): # insert in first 7 random positions
                                    xphrases.insert(rest_all_phrases[g], chunks[g])
                            else:
                                # if remaining chunks are less than 7, insert in between the existing positions and
                                # add the reminder at the end
                                for g in range(len(rest_all_phrases)):
                                    xphrases.insert(rest_all_phrases[g], chunks[g])
                                for g in range(len(rest_all_phrases),6):
                                    xphrases.append(chunks[g])

                            line = " ".join(xphrases)

                        else:
                            line = " ".join(tokLine)

                # log binarised strings
                if outt is not None:
                    outt.write(line.strip() + "\n")

                ids = tokenizer.Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )

                consumer(ids)
                ntok += len(ids)
                ex +=1

            if outt is not None:
                outt.close()

        return {'ntok': ntok,  'nunk': sum(replaced.values()), 'replaced': len(replaced),
                '#examples':ex,
                }


    @staticmethod
    def printStats(res, lang, input_file, dict):

        print('| [{}] {}: {} tokens, {:.3}% replaced by {}'.format(
            lang, input_file, res['ntok'],
            100 * res['nunk'] / res['ntok'], dict.unk_word))
        print("  #examples: {}".format(res['#examples']))