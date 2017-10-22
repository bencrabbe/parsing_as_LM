import os
import os.path
import pickle
import numpy as np
import pandas as pd
import dynet_config
dynet_config.set_gpu()
import dynet as dy
import time

from math import log2,exp
from numpy.random import choice,rand
from lm_utils import RNNLMGenerator
from dataset_utils import ptb_reader,UDtreebank_reader


class RNNLanguageModel:
    """
    A simple RNNLM a la Mikolov 2010.

    Designed to run on a GPU.
    """
    #Undefined and unknown symbols
    UNKNOWN_TOKEN = "<unk>"
    EOS_TOKEN     = "__EOS__" #end  sentence
    IOS_TOKEN   = "__IOS__"   #init sentence 
    
    def __init__(self,embedding_size=300,hidden_size=300,tiedIO = True):
        """
        @param embedding_size: the size of the embedding vectors
        @param lexicon_size  : the number of distinct lexical items
        @param tiedIO : input embeddings are tied with output weights
        """
        self.model = dy.ParameterCollection()
        self.embedding_size   = embedding_size
        self.hidden_size      = hidden_size
        
        self.lexicon_size     = 0    #TBD at train time or at param loading        
        self.word_codes       = None  #TBD at train time or at param loading
        self.rev_word_codes   = None  #TBD at train time or at param loading
        self.tied             = tiedIO
        if self.tied:
            assert(self.embedding_size == self.hidden_size)

    def __str__(self):
        s = ['Embedding size    : %d'%(self.embedding_size),\
            'Hidden layer size : %d'%(self.hidden_size),\
            'Lexicon size      : %d'%(self.lexicon_size),\
            'Tied I/O          : %r'%(self.tied)]
        return '\n'.join(s)


    def read_glove_embeddings(self,glove_filename):
        """
        Reads embeddings from a glove filename and returns an embedding
        matrix for the parser vocabulary.
        @param glove_filename: the file where to read embeddings from
        @return an embedding matrix that can initialize an Embedding layer
        """
        print('Reading embeddings from %s ...'%glove_filename)

        embedding_matrix = (rand(self.lexicon_size,self.embedding_size) - 0.5)/10.0 #an uniform initializer 

        istream = open(glove_filename)
        for line in istream:
            values = line.split()
            word = values[0]
            widx = self.word_codes.get(word)
            if widx != None:
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_matrix[widx] = coefs
        istream.close()
        print('done.')
        return embedding_matrix    

    def code_symbols(self,treebank):
        """
        Codes lexicon (x-data) on integers.
        @param treebank: the treebank where to extract the data from
        """
        lexicon = set([RNNLanguageModel.UNKNOWN_TOKEN,RNNLanguageModel.IOS_TOKEN,RNNLanguageModel.EOS_TOKEN])
        for sentence in treebank:
            lexicon.update(sentence)
        self.lexicon_size = len(lexicon)
        self.rev_word_codes = list(lexicon)
        self.word_codes = dict([(s,idx) for (idx,s) in enumerate(self.rev_word_codes)])

    def make_data_generator(self,treebank,batch_size):
        """
        This returns a data generator suitable for use with dynet
        @param treebank: the treebank (list of sentences) to encode
        @param batch_size: the size of the batches yielded by the generator
        """
        Y  = []
        X  = []
        unk_code = self.word_codes[RNNLanguageModel.UNKNOWN_TOKEN]
        for line in treebank:
            tokens = [RNNLanguageModel.IOS_TOKEN]+line+[RNNLanguageModel.EOS_TOKEN]
            X.append([self.word_codes.get(tok,unk_code) for tok in tokens[:-1]])
            Y.append([self.word_codes.get(tok,unk_code) for tok in tokens[1:]] )
        return RNNLMGenerator(X,Y,self.word_codes[RNNLanguageModel.UNKNOWN_TOKEN],batch_size)

    
    def predict_logprobs(self,X,Y):
        """
        Returns the log probabilities of the predictions for this model (batched version).
        @param X: the input indexes from which to predict (each xdatum is expected to be an iterable of integers) 
        @param Y: a list of references indexes for which to extract the prob. 
        @return the list of predicted logprobabilities for each of the provided ref y in Y
        """
        assert(len(X) == len(Y))

        preds = []

        if self.tied:
            dy.renew_cg()
            state =  self.rnn.initial_state()
            E = dy.parameter(self.embedding_matrix)
            for x,y in zip(X,Y):
                state  = state.add_input(dy.pick(E,x))
                ypred = dy.pickneglogsoftmax(E * state.output(),y)
                preds.append(ypred)
            dy.forward(preds)
            return [-ypred.value() for ypred in preds]
        else:
            dy.renew_cg()
            state =  self.rnn.initial_state()
            O = dy.parameter(self.output_weights)
            E = dy.parameter(self.embedding_matrix)
            for x,y in zip(X,Y):
                state  = state.add_input(dy.pick(E,x))
                ypred = dy.pickneglogsoftmax(O * state.output(),y)
                preds.append(ypred)
            dy.forward(preds)
            return [-ypred.value() for ypred in preds]
    
    def predict_sentence(self,sentence,unk_flag=True,surprisal=True):
        """
        Outputs a sentence together with its predicted transitions probs as a pandas data frame
        @param sententce : a list of strings (words)
        @param unk_flag  : flags if this word is UNK TOKEN
        @param surprisals: also outputs -log2(p)
        @return a pandas DataFrame
        """
        tokens = [NNLanguageModel.IOS_TOKEN]+sentence
        X = [self.word_codes[elt] for elt in tokens[:-1]]
        Y = [self.word_codes[elt] for elt in tokens[1:]]
        
        preds = self.predict_logprobs(X,Y)

        records = []
        cols = ['token','cond_prob']
        if unk_flag:
            cols.append('unk_word')
        if surprisal:
            cols.append('surprisal')
          
        for word,logpred in zip(sentence,preds):
            r = [ word, exp(logpred) ]
            if unk_flag:
                r.append( word not in self.word_codes )
            if surprisal:
                r.append( -logpred)
            records.append(tuple(r))
        return pd.DataFrame(records,columns=cols)


    def train_rnn_lm(self,\
                    train_sentences,\
                    validation_sentences,\
                    lr=0.001,\
                    hidden_dropout=0.1,\
                    rnn_layers=1,\
                    batch_size=100,\
                    max_epochs=100,\
                    glove_file=None):
        """
        Locally trains a model with a static oracle and a standard feedforward NN.  
        @param train_sentences        : a list of sentences
        @param validation_sentences   : a list of sentences
        @return learning curves for various metrics as a pandas dataframe
        """
        #(1) build dictionaries
        self.code_symbols(train_sentences) 
        print("Dictionaries built.")

        #(2) read off treebank and builds data set
        print("Encoding dataset from %d sentences."%len(train_sentences))
        training_generator = self.make_data_generator(train_sentences,batch_size)
        validation_generator = self.make_data_generator(validation_sentences,batch_size)
        
        print(self,flush=True)
        print("max_epochs = %d\ntraining sentences [N] = %d\nBatch size = %d\nDropout = %f\nlearning rate = %f"%(max_epochs,training_generator.get_num_sentences(),batch_size,hidden_dropout,lr),flush=True)

         #(3) Model structure
        self.model = dy.ParameterCollection()
        if glove_file is None:
            self.embedding_matrix = self.model.add_parameters((self.lexicon_size,self.embedding_size))
        else:
            self.embedding_matrix = self.model.parameters_from_numpy(self.read_glove_embeddings(glove_file))
        if not self.tied:
            self.output_weights =  self.model.add_parameters((self.lexicon_size,self.hidden_size))
        self.rnn = dy.LSTMBuilder(rnn_layers, self.embedding_size, self.hidden_size,self.model)

        #fitting
        xgen    =  training_generator.next_batch()
        trainer = dy.AdamTrainer(self.model,alpha=lr)
        min_nll = float('inf')
        history_log = []
        for e in range(max_epochs):
            L = 0
            N = 0
            start_t = time.time()
            for b in range(training_generator.get_num_batches()):
                X,Y = next(xgen)
                if self.tied:
                    dy.renew_cg()
                    state = self.rnn.initial_state()
                    E = dy.parameter(self.embedding_matrix)
                    preds = []
                    for x,y in zip(X,Y):
                        state = state.add_input(dy.pick(E,x))
                        ypred = dy.pickneglogsoftmax(E * state.output(),y)
                        preds.append(ypred)
                    loss = dy.esum(preds)
                    L+= loss.value()
                    loss.backward()
                    trainer.update()
                    N+= len(Y)
                else:
                    dy.renew_cg()
                    state = rnn.initial_state()
                    O = dy.parameter(self.output_weights)
                    E = dy.parameter(self.embedding_matrix)
                    preds = []
                    for x,y in zip(X,Y):
                        state = state.add_input(dy.pick(E,x))
                        ypred = dy.pickneglogsoftmax(O * state.output(),y)
                        preds.append(ypred)
                    loss = dy.esum(preds)
                    L+= loss.value()
                    loss.backward()
                    trainer.update()
                    N+= len(Y)
            end_t = time.time()
            vgen = validation_generator.next_sentence()
            valid_nll = 0
            vN = 0
            for _ in range( validation_generator.get_num_sentences()):
                X,Y = next(vgen)
                valid_nll -= sum(self.predict_logprobs(X,Y))
                vN += len(X)
            valid_ppl = exp(valid_nll/vN)
            history_log.append((e,end_t-start_t,L,exp(L/N),valid_nll,valid_ppl))
            print('Epoch %d (%.2f sec.) NLL (train) = %f, PPL (train) = %f, NLL(valid) = %f, PPL(valid) = %f'%tuple(history_log[-1]),flush=True)
            if valid_nll == min(valid_nll,min_nll):
                min_nll = valid_nll
                #self.save_model('best_model_dump',epoch=e)
        return pd.DataFrame(history_log,columns=['epoch','wall_time','NLL(train)','PPL(train)','NLL(dev)','PPL(dev)'])

if __name__ == '__main__':

    #read penn treebank
    ttreebank =  ptb_reader('ptb/ptb_train_50w.txt')
    dtreebank =  ptb_reader('ptb/ptb_valid.txt')
    
    lm = RNNLanguageModel(hidden_size=300,embedding_size=300,tiedIO=True)
    lm.train_rnn_lm(ttreebank[:10],dtreebank[:10],lr=0.1,hidden_dropout=0.3,batch_size=512,max_epochs=20,glove_file='glove/glove.6B.300d.txt')
    #lm.save_model('final_model')
