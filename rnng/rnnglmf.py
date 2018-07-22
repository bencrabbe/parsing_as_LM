#!/usr/bin/env python

import numpy as np
import dynet as dy
import getopt
import json

from random import shuffle
from lexicons import *

"""
That's a simple RNNLM designed to be interoperable with the rnng parser.
Allows comparisons and sharing of parameters.
** It is designed to run mainly on a CPU **
"""

class RNNGlm:
    
    #special tokens
    UNKNOWN_TOKEN = '<UNK>'
    START_TOKEN   = '<START>'

    def __init__(self,brown_clusters,max_vocabulary_size=10000,embedding_size=50,memory_size=50):
        """
        @param brown_clusters          : a filename where to find brown clusters
        @param max_vocabulary_size     : max number of words in the vocab
        @param stack_embedding_size    : size of stack lstm input 
        @param stack_memory_size       : size of the stack and tree lstm hidden layers
        """
        self.max_vocab_size = max_vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size    = memory_size
        self.dropout        = 0.0
        self.brown_file     = brown_clusters
        
        #Extras (brown lexicon and external embeddings)
        self.ext_embeddings = False

    def code_lexicon(self,raw_treebank):
        
        lexicon = Counter()
        for sentence in raw_treebank:
            lexicon.update(sentence)
        known_vocabulary = set([word for word, counts in lexicon.most_common(self.max_vocab_size)])
        known_vocabulary.add(RNNGlm.START_TOKEN)
        
        self.brown_file  = normalize_brown_file(self.brown_file,known_vocabulary,self.brown_file+'.unk',UNK_SYMBOL=RNNGlm.UNKNOWN_TOKEN)
        self.lexicon     = SymbolLexicon( list(known_vocabulary),unk_word=RNNGlm.UNKNOWN_TOKEN)
   
    def make_structure(self):
        """
        Creates and allocates the network structure
        """
        #Model structure
        self.model = dy.ParameterCollection()

        #Lex input
        self.E    = self.model.add_lookup_parameters((self.lexicon.size(),self.embedding_size))
        #Lex output
        self.O    = dy.ClassFactoredSoftmaxBuilder(self.hidden_size,self.brown_file,self.lexicon.words2i,self.model,bias=True)
        #RNN
        self.rnn = dy.LSTMBuilder(1,self.embedding_size,self.hidden_size,self.model)  

    def rnn_dropout(self,expr):
        """
        That is a conditional dropout that applies dropout to a dynet expression only at training time
        @param expr: a dynet expression
        @return a dynet expression
        """
        if self.dropout == 0:
            return expr
        else:
            return dy.dropout(expr,self.dropout)

        
    def train_rnn_lm(self,modelname,train_sentences,validation_sentences,lr=0.1,dropout=0.3,max_epochs=10,batch_size=1):
        """
        Trains an RNNLM on a data set. Vanilla SGD training mode with simple minibatching and without any funny optimization.
        @param modelname: a string used as prefix of output files
        @param train_sentences,validation_sentences: lists of strings
        @param lr: learning rate for SGD
        @param dropout: dropout
        @param max_epochs : number of epochs to run
        @param batch_size : size of batches
        """

        self.code_lexicon(train_sentences)
        self.make_structure()
        self.print_summary()

        trainer = dy.SimpleSGDTrainer(self.model,learning_rate=lr)
        min_nll = np.inf

        ntrain_sentences = len(train_sentences)
        ndev_sentences   = len(train_sentences)
        
        for e in range(max_epochs):

            NLL = 0
            N = 0
            
            batches_processed = 0
            bbegin = 0
            
            while bbegin < ntrain_sentences:
                dy.renew_cg()
                outputs = []
                bbegin = batches_processed * batch_size
                bend = min(ntrain_sentences,bbegin + batch_size)
                print(bbegin,bend)
                for sent in train_sentences[bbegin:bend]:
                    X          = [self.lexicon.index(word) for word  in [RNNGlm.START_TOKEN] + sent[:-1] ]
                    Y          = [self.lexicon.index(word) for word in sent]
                    state      = self.rnn.initial_state()
                    xinputs    = [dy.dropout(self.E[x],self.dropout) for x in X]
                    state_list = state.add_inputs(xinputs)
                    outputs.extend([self.O.neg_log_softmax(dy.dropout(S.output(),self.dropout),y) for (S,y) in zip(state_list,Y) ])
                    N         += len(Y)

                loc_nll    = dy.esum(outputs)
                NLL       += loc_nll.value()
                loc_nll.backward()
                trainer.update()
                batches_processed += 1
                    
            print('[Training]   Epoch %d, NLL = %f, PPL = %f'%(e,NLL,np.exp(NLL/N)),flush=True)

            NLL = 0
            N = 0
            
            batches_processed = 0
            bbegin = 0
            while bbegin < ntrain_sentences:
                dy.renew_cg()
                outputs = []                
                bbegin = batches_processed * batch_size
                bend = min(ntrain_sentences,bbegin + batch_size)
                for sent in validation_sentences[bbegin:bend]:
                    X          = [self.lexicon.index(word) for word  in [RNNGlm.START_TOKEN] + sent[:-1] ]
                    Y          = [self.lexicon.index(word) for word in sent]
                    state      = self.rnn.initial_state()
                    xinputs    = [self.E[x] for x in X]
                    state_list = state.add_inputs(xinputs)
                    outputs.extend([self.O.neg_log_softmax(S.output(),y) for (S,y) in zip(state_list,Y) ])
                    N         += len(Y)
                loc_nll    = dy.esum(outputs)
                NLL       += loc_nll.value()
                batches_processed += 1
 
            print('[Validation] Epoch %d, NLL = %f, PPL = %f\n'%(e,NLL,np.exp(NLL/N)),flush=True)


            
    def print_summary(self):
        """
        Prints a summary of the model structure
        """
        print('Lexicon size            :',self.lexicon.size(),flush=True)
        print('embedding size          :',self.embedding_size,flush=True)
        print('hidden size             :',self.hidden_size,flush=True)


if __name__ == '__main__':
    
    istream  = open('ptb_train.raw')
    train_treebank = [line.split() for line in istream]
    istream.close()

    istream  = open('ptb_dev.raw')
    dev_treebank = [line.split() for line in istream]
    istream.close()
    
    rnnlm = RNNGlm('ptb-250.brown',embedding_size=50,memory_size=50)
    rnnlm.train_rnn_lm('testlm',train_treebank,dev_treebank,lr=0.1,dropout=0.3,max_epochs=20,batch_size=32)    



        
