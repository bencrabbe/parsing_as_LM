#!/usr/bin/env python

import numpy as np
import dynet as dy
import getopt
import json
import sys

from random import shuffle
from lexicons import *

"""
That's a simple class factored RNNLM designed to be interoperable with the rnng parser.
Allows comparisons.
** It is designed to run mainly on a CPU **
"""

class RNNGlm:
    
    #special tokens
    UNKNOWN_TOKEN = '<UNK>'
    START_TOKEN   = '<START>'
    
    def __init__(self,brown_clusters,vocab_thresh=1,embedding_size=50,memory_size=50):
        """
        @param brown_clusters          : a filename where to find brown clusters
        @param vocab_thresh            : number of counts above which a word is known to the vocab
        @param stack_embedding_size    : size of stack lstm input 
        @param stack_memory_size       : size of the stack and tree lstm hidden layers
        """
        self.vocab_thresh   = vocab_thresh
        self.embedding_size = embedding_size
        self.hidden_size    = memory_size
        self.dropout        = 0.0
        self.brown_file     = brown_clusters
        
        #Extras (brown lexicon and external embeddings)
        self.ext_embeddings = False

    def code_lexicon(self,raw_treebank):
        
        known_vocabulary = get_known_vocabulary(raw_treebank,vocab_treshold=1)
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

        trainer = dy.SimpleSGDTrainer(self.model,learning_rate=lr)
        min_nll = np.inf

        ntrain_sentences = len(train_sentences)

        self.print_summary(ntrain_sentences,len(validation_sentences),lr,dropout)

        for e in range(max_epochs):

            NLL = 0
            N = 0
            
            bbegin = 0
            
            while bbegin < ntrain_sentences:
                dy.renew_cg()
                outputs = []
                bend = min(ntrain_sentences,bbegin + batch_size)
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
                bbegin = bend
                    
            print('[Training]   Epoch %d, NLL = %f, PPL = %f'%(e,NLL,np.exp(NLL/N)),flush=True)

            NLL,N = self.eval_model(validation_sentences,batch_size)
                
            if NLL < min_nll:
                self.save_model(modelname)
                min_nll = NLL
                
            print('[Validation] Epoch %d, NLL = %f, PPL = %f\n'%(e,NLL,np.exp(NLL/N)),flush=True)

            
    def eval_model(self,test_sentences,batch_size,stats_file=None):
        """
        Tests a model on a validation set and returns the NLL and the Number of words in the dataset.
        Args:
             test_sentences (list): a list of list of strings.
             batch_size      (int): the size of the batch used
        Kwargs:
             stats_file   (stream): the stream where to write the stats or None
        Returns:
             a couple that allows to compute perplexities. (Negative LL,N) 
        """
        
        NLL = 0
        N = 0
        ntest_sentences   = len(test_sentences)

        batches_processed = 0
        bbegin = 0

        stats_header = True 
        while bbegin < ntest_sentences:
            dy.renew_cg()
            outputs = []                
            bend = min(ntest_sentences,bbegin + batch_size)
            for sent in test_sentences[bbegin:bend]:
                X          = [self.lexicon.index(word) for word  in [RNNGlm.START_TOKEN] + sent[:-1] ]
                Y          = [self.lexicon.index(word) for word in sent]
                state      = self.rnn.initial_state()
                xinputs    = [self.E[x] for x in X]
                state_list = state.add_inputs(xinputs)
                outputs.extend([self.O.neg_log_softmax(S.output(),y) for (S,y) in zip(state_list,Y) ])
                N         += len(Y)
            loc_nll    = dy.esum(outputs)
            NLL       += loc_nll.value()

            if stats_file:###stats generation
                if stats_header:
                    print('token\tcond_logprob\tsurprisal\tis_unk',file=stats_file)
                toklist = [] 
                for sent in test_sentences[bbegin:bend]:
                    toklist.extend(sent)
                batch_stats  = '\n'.join(["%s\t%f\t%f\t%s"%(word,logprob.value(),-logprob.value()/np.log(2),word in self.lexicon) for word,logprob in zip(toklist,outputs)])
                print(batch_stats,file=stats_file)
                stats_header = False
                
            batches_processed += 1
            bbegin = batches_processed * batch_size

        return (NLL,N)

    
    def print_summary(self,ntrain,ndev,lr,dropout):
        """
        Prints a summary of the model structure.
        """
        print('# Training sentences    :',ntrain)
        print('# Validation sentences  :',ndev)
        print('Lexicon size            :',self.lexicon.size(),flush=True)
        print('embedding size          :',self.embedding_size,flush=True)
        print('hidden size             :',self.hidden_size,flush=True)
        print('Learning rate           :',lr,flush=True)
        print('Dropout                 :',dropout,flush=True)

    def save_model(self,model_name):
        """
        Saves the whole shebang.
        """
        jfile = open(model_name+'.json','w')        
        jfile.write(json.dumps({'vocab_thresh'    :self.vocab_thresh,\
                                'embedding_size':self.embedding_size,\
                                'brown_file':self.brown_file,\
                                'hidden_size':self.hidden_size}))
        self.model.save(model_name+'.prm')
        self.lexicon.save(model_name+'.lex')

    @staticmethod
    def load_model(model_name):
        """
        Loads the whole shebang and returns an LM.
        """
        struct     = json.loads(open(model_name+'.json').read())
        lm         = RNNGlm(struct['brown_file'],vocab_thresh=struct['vocab_thresh'],embedding_size=struct['embedding_size'],memory_size=struct['hidden_size'])
        lm.lexicon = SymbolLexicon.load(model_name+'.lex')
        lm.make_structure()
        lm.model.populate(model_name+".prm")
        return lm

def read_config(filename=None):

    """
    Return an hyperparam dictionary
    """
    import configparser
    config = configparser.ConfigParser()
    config.read(filename)

    params = {}
    params['embedding_size'] = int(config['structure']['embedding_size']) if 'embedding_size' in config['structure'] else 100
    params['memory_size']    = int(config['structure']['memory_size'])    if 'memory_size' in config['structure'] else 100
    params['dropout']         = float(config['learning']['dropout'])      if 'dropout' in config['learning'] else 0.1
    params['learning_rate']   = float(config['learning']['learning_rate'])if 'learning_rate' in config['learning'] else 0.1
    params['num_epochs']      = int(config['learning']['num_epochs'])     if 'num_epochs' in config['learning'] else 20
    return params


    
if __name__ == '__main__':
    
    train_file  = ''
    dev_file    = ''
    test_file   = ''
    brown_file  = ''
    model_name  = ''
    config_file = ''
    stats       = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:d:p:m:b:c:s")
    except getopt.GetoptError:
        print('Ooops, wrong command line arguments')
        print('for training...')
        print ('rnnglm.py -t <inputfile> -d <inputfile> -m <model_file> -b <brown_file>')
        print('for testing...')
        print ('rnnglm.py -m <model_file> -p <test_file>')
        sys.exit(0)
        
    for opt, arg in opts:
        if opt in ['-t','--train']:
            train_file = arg
        elif opt in ['-d','--dev']:
            dev_file   = arg
        elif  opt in ['-p','--pred']:
            test_file = arg
        elif opt in  ['-m','--model']:
            model_name = arg
        elif opt in  ['-b','--brown']:
            brown_file = arg
        elif opt in ['-c','--config']:
            config_file = arg
        elif opt in ['-s','--stats']:
            stats_file = arg
        else:
            print('unknown option %s, ignored'%(arg))
            
    if train_file and dev_file and brown_file and model_name:
        
        istream  = open(train_file)
        train_treebank = [line.split() for line in istream]
        istream.close()

        istream  = open(dev_file)
        dev_treebank = [line.split() for line in istream]
        istream.close()
        

        if config_file:
            config = read_config(config_file)
            rnnlm = RNNGlm(brown_file,embedding_size=config['embedding_size'] ,memory_size=config['memory_size'])
            rnnlm.train_rnn_lm(model_name,train_treebank,dev_treebank,lr=config['learning_rate'],dropout=config['dropout'],max_epochs=config['num_epochs'],batch_size=32)
        else:
            rnnlm = RNNGlm(brown_file,embedding_size=100 ,memory_size=100)
            rnnlm.train_rnn_lm(model_name,train_treebank,dev_treebank,lr=0.1,dropout=0.5,max_epochs=40,batch_size=32)

        print('training done.')
        
    if model_name and test_file:

        rnnlm = RNNGlm.load_model(model_name+'/'+model_name)

        istream       = open(test_file)
        test_treebank = [line.split() for line in istream]
        istream.close()

        stats_stream = open(model_name + '/' + model_name +'.tsv','w') if stats else None
        print('Test PPL',rnnlm.eval_model(test_treebank,batch_size=32,stats_file=stats_stream))
        if stats:
            stats_stream.close()
