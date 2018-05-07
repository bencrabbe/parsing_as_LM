#!/usr/bin/env python

import numpy as np
import numpy.random as npr
#import dynet_config
#dynet_config.set_gpu()
import dynet as dy
import getopt
import json
import pandas as pd

from collections import Counter
from random import shuffle
from lexicons import *
from lm_utils import *

"""
That's an RNNLM designed to be interoperable with the rnng parser.
Allows comparisons and sharing of parameters.
It is designed to run on a GPU.
"""

class RNNGlm:

    #special tokens
    UNKNOWN_TOKEN = '<UNK>'
    START_TOKEN   = '<START>'    

    def __init__(self,max_vocabulary_size=10000,embedding_size=50,memory_size=50):
        """
        @param max_vocabulary_size     : max number of words in the vocab
        @param stack_embedding_size    : size of stack lstm input 
        @param stack_memory_size       : size of the stack and tree lstm hidden layers
        """
        self.max_vocab_size = max_vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size    = memory_size
        self.dropout        = 0.0
        
        #Extras (brown lexicon and external embeddings)
        self.blex           = None
        self.ext_embeddings = False
        #self.tied           = False

                
    def code_lexicon(self,raw_treebank,max_vocab_size):
        
        #normal lexicon
        lexicon = Counter()
        for sentence in raw_treebank:
            lexicon.update(sentence)
        self.lexicon = SymbolLexicon(lexicon,unk_word=RNNGlm.UNKNOWN_TOKEN,special_tokens=[RNNGlm.START_TOKEN],max_lex_size=10000)
    

    def lex_lookup(self,token):
        """
        Performs lookup and backs off unk words to the unk token
        @param token : the string token to code
        @return : word_code for in-vocab tokens and word code of unk word string for OOV tokens
        """
        return self.lexicon.index(token)
    
    def cls_lookup(self,token): 
        """
        Performs lookup for clusters
        @param token : the string token for which to find the cluster idx
        @return : cluster code for in-vocab tokens and cluster code of unk words for OOV tokens
        """
        return self.blex.index(token)


    #scoring & representation system
    def make_structure(self,w2vfilename=None):
        """
        Allocates the network structure
        @param w2filename: an external word embedding dictionary
        """
        #Model structure
        self.model                 = dy.ParameterCollection()

        #input embeddings 
        if not w2vfilename:
            self.lex_embedding_matrix  = self.model.add_parameters((self.lexicon.size(),self.embedding_size))  
        else:
            print('Using external embeddings.',flush=True)                                                          #word embeddings
            self.ext_embeddings       =  True

            W,M = RNNGlm.load_embedding_file(w2vfilename)
            embed_dim = M.shape[1]
            self.embedding_size = embed_dim
            E = self.init_ext_embedding_matrix(W,M)
            self.lex_embedding_matrix = self.model.parameters_from_numpy(E) 

        if self.blex:
            self.lex_out            = self.model.add_parameters((self.blex.size(),self.hidden_size))                           #lex action output layer
            self.lex_bias           = self.model.add_parameters(self.blex.size())
        else:
            self.lex_out            = self.model.add_parameters((self.lexicon.size(),self.hidden_size))                       #lex action output layer
            self.lex_bias           = self.model.add_parameters(self.lexicon.size())

        #rnn 
        self.rnn                    = dy.LSTMBuilder(1,self.embedding_size,self.hidden_size,self.model)                # main rnn

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

    def rnn_nobackprop(self,expr,word_token):
       """
       Function controlling whether to block backprop or not on lexical embeddings
       @param expr: a dynet expression
       @param word_token: the lexical token 
       @return a dynet expression
       """
       if self.ext_embeddings and word_token in self.word_codes:  #do not backprop with external embeddings when word is known to the lexicon
           return dy.nobackprop(expr)
       else:
           return expr
   
    def make_data_generator(self,raw_treebank,batch_size):
        """
        This returns a data generator suitable for use with dynet
        @param raw_treebank: the treebank (list of sentences) to encode
        @param batch_size: the size of the batches yielded by the generator
        @return a data generator that yields integer encoded batches
        """
        Y  = []
        X  = []
        for line in raw_treebank:
            tokens = [RNNGlm.START_TOKEN]+line
            X.append([self.lexicon.index(tok) for tok in tokens[:-1]])
            if self.blex:
                Y.append([self.blex.index(tok) for tok in tokens[1:]] )
            else:
                Y.append([self.lexicon.index(tok) for tok in tokens[1:]] )
        if self.blex:
            return RNNLMGenerator(X,Y,self.blex.index(RNNGlm.START_TOKEN),batch_size)
        else:
            return RNNLMGenerator(X,Y,self.lexicon.index(RNNGlm.START_TOKEN),batch_size)


    def train_rnn_lm(self,modelname,train_sentences,validation_sentences,lr=0.0001,dropout=0.3,batch_size=100,max_epochs=100,cls_filename=None,w2v_file=None):

        self.dropout = dropout
        
        #coding
        if cls_filename:
            print("Using clusters")
            self.blex = BrownLexicon.read_clusters(cls_filename,freq_thresh=1,UNK_SYMBOL=RNNGlm.UNKNOWN_TOKEN)

        self.code_lexicon(train_sentences,self.max_vocab_size)
        #structure
        self.make_structure(w2v_file)

        self.print_summary()
        print(len(train_sentences))
        #coding dataset & batching        
        training_generator = self.make_data_generator(train_sentences,batch_size)
        xgen    =  training_generator.next_batch()

        #trainer = dy.RMSPropTrainer(self.model,learning_rate=lr)
        trainer = dy.SimpleSGDTrainer(self.model,learning_rate=lr)
        min_nll = np.inf
        for e in range(max_epochs):
            L = 0
            N = 0
            for b in range(training_generator.get_num_batches()):
                X,Y = next(xgen)                          #all batch elts are guaranteed to have equal size.
                time_steps = len(X[0])
                
                X,Y = list(zip(*X)),list(zip(*Y))         #transposes the batch

                losses     = [ ]
                
                dy.renew_cg()
                O = dy.parameter(self.lex_out)
                b = dy.parameter(self.lex_bias)
                E = dy.parameter(self.lex_embedding_matrix)
                
                state = self.rnn.initial_state()
                if self.ext_embeddings:
                    lookups    = [ dy.nobackprop(dy.dropout(dy.pick_batch(E,xcolumn),self.dropout)) for xcolumn in X ]
                else:
                    lookups    = [ dy.dropout(dy.pick_batch(E,xcolumn),self.dropout) for xcolumn in X ]
                outputs    = state.transduce(lookups)
                losses     = [ dy.pickneglogsoftmax_batch(O * dy.dropout(lstm_out,self.dropout)+ b ,y) for lstm_out,y in zip(outputs,Y) ]
                batch_loss = dy.sum_batches(dy.esum(losses))
                L  +=  batch_loss.value()
                batch_loss.backward()
                trainer.update()
                N         += sum( [ len(row)  for row in Y     ] )
                
            print('train (optimistic)','Mean NLL',L/N,'PPL',np.exp(L/N))
            eL,eN = self.eval_lm(validation_sentences)
            print('eval ','Mean NLL',eL/eN,'PPL',np.exp(eL/eN))
            if eL <= min_nll :
                min_nll= eL
                print(" => saving model",eL)
                #self.save_model(modelname)

    def save_model(self,model_name):
        """
        Saves the whole shebang.
        """
        #TODO (with updated lexicon)
        jfile = open(model_name+'.json','w')
        jfile.write(json.dumps({'embedding_size':self.embedding_size,\
                                'hidden_size':self.hidden_size}))
                                
        self.model.save(model_name+'.prm')
        self.lexicon.save(model_name+'.lex')
        if self.blex:
            self.blex.save_clusters(model_name+'.cls')
            
    def load_model(self,modelname):
        """
        Loads the whole shebang and returns an LM.
        """
        #TODO (with updated lexicon)
        struct = json.loads(open(model_name+'.json').read())
        lm         = RNNGlm(embedding_size = struct['embedding_size'],memory_size = struct['hidden_size'])
        lm.lexicon = SymbolLexicon.load(modelname+'.lex')
        try:
            lm.blex = BrownLexicon.load_clusters(model_name+'.cls')
        except FileNotFoundError:
            print('No clusters found',file=sys.stderr)
            self.blex = None
        lm.make_structure()
        lm.model.populate(model_name+".prm")
        return lm
    
    def eval_lm(self,sentences):
        """
        Evaluates a model sentence by sentence.
        Inefficient but exact method, including with clusters.
        @param sentences : a list of list of words
        @return : a couple (negative log likelihood,perplexity) 
        """
        N = 0
        L = 0
        for tokens in sentences:
            tokens  = [RNNGlm.START_TOKEN] + tokens
            x_codes = [self.lexicon.index(tok) for tok in tokens[:-1]]
            y_codes = [self.blex.index(tok) for tok in tokens] if self.blex else [self.lexicon.index(tok) for tok in tokens[1:]] 

            dy.renew_cg()
            O = dy.parameter(self.lex_out)
            b = dy.parameter(self.lex_bias)
            E = dy.parameter(self.lex_embedding_matrix)
                
            state = self.rnn.initial_state()
            lookups    = [ dy.pick(E,xcolumn) for xcolumn in x_codes ]
            outputs    = state.transduce(lookups)
            for tok,lstm_pred,yref in zip (tokens[1:],outputs,y_codes):
                loss     = dy.pickneglogsoftmax(O * lstm_pred + b, yref).value() 
                L       += loss 
                if self.blex:
                    L += self.blex.word_emission_prob(tok)
                N += 1
        return (L,N)
    
   
                
    #I/O etc.
    def print_summary(self):
        """
        Prints a summary of the parser structure
        """
        lexA =   self.blex.size() if self.blex else self.lexicon.size()
        print('Num Lexical actions     :',lexA,flush=True)
        print('Lexicon size            :',self.lexicon.size(),flush=True)
        print('embedding size          :',self.embedding_size,flush=True)
        print('hidden size             :',self.hidden_size,flush=True)

    @staticmethod
    def load_embedding_file(w2vfilename):
        """
        Reads a word2vec file and returns a couple (list of strings, matrix of vectors)
        @param w2vfilename : the word2 vec file (Mikolov format)
        @return (wordlist,numpy matrix)
        """
        istream = open(w2vfilename)
        wordlist = []
        veclist  = []
        istream.readline()#skips header
        for line in istream:
            fields = line.split()
            wordlist.append(fields[0])            
            veclist.append(np.asarray(fields[1:],dtype='float32'))
        M = np.array(veclist)
        #print(M[:3,:])
        return (wordlist, M)

    def init_ext_embedding_matrix(self,emb_wordlist,matrix):
        """
        Initializer for external embedding matrix.
        
        Returns numpy matrix ready to use as initializer of a dynet param
        @param emb_wordlist: the nomenclature of external embeddings
        @param matrix: the related embedding vectors
        @return a matrix ready to initalize the dynet params
        """
        r = self.lexicon.size()
        c = matrix.shape[1]
        new_mat = npr.randn(r,c)/100 #gaussian init with small variance (applies for unk words)
        for emb_word,emb_vec in zip(emb_wordlist,matrix):
            idx = self.lexicon.index(emb_word)
            if idx != self.lexicon.get_UNK_ID():
                new_mat[idx,:] = emb_vec                
        return new_mat
       

if __name__ == '__main__':

    istream  = open('ptb_train.raw')
    train_treebank = [line.split() for line in istream]
    istream.close()

    istream  = open('ptb_dev.raw')
    dev_treebank = [line.split() for line in istream]
    istream.close()

    rnnlm = RNNGlm(embedding_size=100,memory_size=300)

    #Good cluster model PPL 130 on devel
    #rnnlm.train_rnn_lm('testlm',train_treebank,dev_treebank,lr=0.0001,dropout=0.3,batch_size=32,max_epochs=100,cls_filename='ptb-1000.brown',w2v_file='word_embeddings/w2v-ptb.txt')

    rnnlm.train_rnn_lm('testlm',train_treebank,dev_treebank,lr=0.001,dropout=0.4,batch_size=32,max_epochs=200,cls_filename='ptb-250.brown')    

    #Good word model : PPL on devel
    #rnnlm.train_rnn_lm('testlm',train_treebank,dev_treebank,lr=0.001,dropout=0.3,batch_size=32,max_epochs=50,w2v_file='word_embeddings/w2v-ptb.txt')    

    #rnnlm.train_rnn_lm('testlm',train_treebank,dev_treebank,lr=0.001,dropout=0.3,batch_size=100,max_epochs=15,w2v_file='word_embeddings/w2v-ptb.txt')    


    
        



    

    
    
