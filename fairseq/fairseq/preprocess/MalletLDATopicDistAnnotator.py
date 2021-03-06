
import operator
import torch
import gensim

from .LDATopicDistAnnotator import LDATopicDistAnnotator, stop_words

class MalletLDATopicDistAnnotator(LDATopicDistAnnotator):

    def loadModel(self, ldaModel, bigram_mod=None):
        print("Load Mallet LDA model")
        self.lda_mod = gensim.models.wrappers.LdaMallet.load(ldaModel)
        self.bigram = bigram_mod is not None
        if self.bigram:
            self.bigram_mod = gensim.models.phrases.Phraser.load(bigram_mod)

    def generateKeywordEmbeddings(self, save_dir):

        topic_term_matrix = self.lda_mod.load_word_topics()

        f = open(save_dir, 'w', encoding='utf8')
        for term in self.keyvocab.keys():
            if term is self.kpad_word:
                continue
            if term is self.outOfTopic:
                continue

            id = self.lda_mod.id2word.token2id[term]
            y = topic_term_matrix[:,id]
            y = y / y.sum()
            f.write(term + " " + " ".join([str(y[i]) for i in range(len(y))]) + "\n")
        f.close()

    def topicDistrib(self, phrase, caller=None):

        def remove_stopwords(texts):
            return [[word for word in doc if word not in stop_words] for doc in texts]

        topic_vector = []

        phrase_lemmatized = remove_stopwords(phrase)

        phrase_bin = [self.lda_mod.id2word.doc2bow(p) for p in phrase_lemmatized] #returns lists of processed texts

        all_doc_topics = self.lda_mod.__getitem__(phrase_bin, iterations=10) # Even with reduced iterations is TOO SLOW! Unusable


        #all_topics = self.lda_mod.get_document_topics(phrase_bin, per_word_topics=True)
        for doc_topics, phph, phphall in zip(all_doc_topics, phrase_lemmatized, phrase):
            x = torch.FloatTensor([t[1] for t in doc_topics] )
            if len(doc_topics) <= self.num_topics:
                x_new = x.new(self.num_topics + 1).fill_(0.0) # first t topics from model, + OtherTopic
                if len(phph)==0 :
                    x_new[self.num_topics] = 1
                else:
                    for j in range(len(doc_topics)):
                        x_new[doc_topics[j][0]] = x[j]
                    #    if x[j] >= 0.2 :
                    #        x_new[doc_topics[j][0]] = 1
                    #if sum(x_new)==0:
                    #    x_new[self.num_topics] = 1 #debugged, works fine

            # debugging
            #if caller is "sent":
            #print("\n\n")
            #print("> ", phphall)
            #print(len(phphall))
            #print(doc_topics)
            #print(x_new)
            #for t,s in doc_topics:
            #    print([w for w in self.topickeys[t] ])
            #print("\n")

            if x.sum().item()<= 0.0:
                print("ERROR!")
                print(doc_topics)
                exit()

            topic_vector.append(x_new)

        print("Did example")
        return topic_vector

    def keywords(self, phrase):
        keyword_vector = []

        def remove_stopwords(texts):
            return [[word for word in doc if word not in stop_words] for doc in texts]

        phrase_lemmatized = remove_stopwords(phrase)
        phrase_bin = [self.lda_mod.id2word.doc2bow(p) for p in phrase_lemmatized] #returns lists of processed texts
        #all_topics = self.lda_mod.get_document_topics(phrase_bin, per_word_topics=True)
        #for (doc_topics, word_topics, phi_values), phph in zip(all_topics, phrase_lemmatized):
        for phi, phph, phphall in zip(phrase_bin, phrase_lemmatized, phrase):
            doc_topics = self.lda_mod[phi]

            doc_topics.sort(key=operator.itemgetter(1), reverse = True)
            if len(doc_topics)==0:
                print("0 doc_topics")
                print(doc_topics)
                exit()
            ###topicJudgement = len(word_topics)>=(len(phph)*0.2) #only use topic labels if most of the words were classified
            ###didnt work

            # take keywords of to two highest scoring topics
            keysid = []
            if doc_topics[0][1] >= 0.2 : #and topicJudgement:
                keysid.extend([self.keyvocab[w] for w in self.topickeys[doc_topics[0][0]] ]) # first one
            else:
                keysid.extend([self.keyvocab[self.outOfTopic]] + ( [self.keyvocab[self.kpad_word]] * (self.num_words-1)))  # first one
            if len(doc_topics)< 2 or doc_topics[1][1] < 0.2 : #or not topicJudgement:
                keysid.extend([self.keyvocab[self.kpad_word]] * self.num_words)
            else:
                keysid.extend([self.keyvocab[w] for w in self.topickeys[doc_topics[1][0]]]) # second if exists
            keyword_vector.append(torch.IntTensor(keysid))

            #print("\n\n")
            #print(phph)
            #print(doc_topics)
            #for t,s in doc_topics:
            #    print([w for w in self.topickeys[t] ])
            #print(topicJudgement)
            #print(keysid)


        return keyword_vector

