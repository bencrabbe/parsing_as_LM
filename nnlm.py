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
from lm_utils import NNLMGenerator
from dataset_utils import ptb_reader,UDtreebank_reader


class NNLanguageModel:
    """
    A simple NNLM a la Bengio 2002.

    Designed to run on a GPU.
    """
    #Undefined and unknown symbols
    UNKNOWN_TOKEN = "<unk>"
    EOS_TOKEN     = "__EOS__" #end  sentence
    IOS_TOKEN   = "__IOS__"   #init sentence 
    
    def __init__(self,\
                 input_length=3,\
                 embedding_size=300,\
                 hidden_size=300,
                 tiedIO = True):
        """
        @param input_length  : the number of x conditioning words in the input
        @param embedding_size: the size of the embedding vectors
        @param lexicon_size  : the number of distinct lexical items
        @param tiedIO : input embeddings are tied with output weights
        """
        self.model = dy.ParameterCollection()
        self.input_length     = input_length    #num symbols fed to the network for predictions
        self.embedding_size   = embedding_size
        self.hidden_size      = hidden_size
        
        self.lexicon_size     = 0    #TBD at train time or at param loading        
        self.word_codes       = None  #TBD at train time or at param loading
        self.rev_word_codes   = None  #TBD at train time or at param loading
        self.tied             = tiedIO
        if self.tied:
            assert(self.embedding_size == self.hidden_size)
        
    def __str__(self):
        s = ['Input Length      : %d'%(self.input_length),\
            'Embedding size    : %d'%(self.embedding_size),\
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
    
    def make_representation(self,xcontext,prediction=None):
        """
        Turns a configuration into a couple of vectors (X,Y) data and
        outputs the coded configuration as a tuple of vectors suitable
        for dynet.
        @param xcontext  : the actual preceding words
        @param prediction: the actual y value (next word) or None in case of true prediction.  
        @return a couple (X,Y) or just X if no prediction is given as param
        """
        Nx   = len(xcontext)
        X    = [self.word_codes[NNLanguageModel.IOS_TOKEN]]*self.input_length
        unk_code = self.word_codes[NNLanguageModel.UNKNOWN_TOKEN]
        #print('unk=',unk_code)
        if  Nx > 0:  
            X[0] = self.word_codes.get(xcontext[-1],unk_code)
        if  Nx > 1:
            X[1] = self.word_codes.get(xcontext[-2],unk_code)
        if Nx > 2:
            X[2] = self.word_codes.get(xcontext[-3],unk_code) 
    
        if prediction is not None:
            Y = self.word_codes.get(prediction,unk_code)
            return (X,Y)
        else:
            return X

    def code_symbols(self,treebank):
        """
        Codes lexicon (x-data) on integers.
        @param treebank: the treebank where to extract the data from
        """
        lexicon = set([NNLanguageModel.UNKNOWN_TOKEN,NNLanguageModel.IOS_TOKEN,NNLanguageModel.EOS_TOKEN])
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
        Y    = []
        X    = []
        for line in treebank:
            tokens = [NNLanguageModel.IOS_TOKEN,NNLanguageModel.IOS_TOKEN,NNLanguageModel.IOS_TOKEN]+line+[NNLanguageModel.EOS_TOKEN]
            for (w3,w2,w1,wy) in zip(tokens,tokens[1:],tokens[2:],tokens[3:]):
                x,y    = self.make_representation([w3,w2,w1],wy)
                X.append(x)
                Y.append(y)
        return NNLMGenerator(X,Y,self.word_codes[NNLanguageModel.UNKNOWN_TOKEN],batch_size)

    
    def predict_logprobs(self,X,Y):
        """
        Returns the log probabilities of the predictions for this model (batched version).

        @param X: the input indexes from which to predict (each xdatum is expected to be an iterable of integers) 
        @param Y: a list of references indexes for which to extract the prob. 
        @return the list of predicted logprobabilities for each of the provided ref y in Y
        """
        assert(len(X) == len(Y))
        assert(all(len(x) == self.input_length for x in X))

        preds = []

        if self.tied:
            dy.renew_cg()
            W = dy.parameter(self.hidden_weights)
            E = dy.parameter(self.embedding_matrix)
            for x,y in zip(X,Y):
                embeddings = [dy.pick(E, widx) for widx in x]
                xdense     = dy.concatenate(embeddings)
                ypred     = dy.pickneglogsoftmax(E * dy.tanh( W * xdense ),y)
                preds.append(ypred)
            dy.forward(preds)
            return [-ypred.value()  for ypred in preds]
        else:
            dy.renew_cg()
            O = dy.parameter(self.output_weights)
            W = dy.parameter(self.hidden_weights)
            E = dy.parameter(self.embedding_matrix)
            for x,y in zip(X,Y):
                embeddings = [dy.pick(E, widx) for widx in x]
                xdense     = dy.concatenate(embeddings)
                ypred      = dy.pickneglogsoftmax(O * dy.tanh( W * xdense ),y)
                preds.append(ypred)
            dy.forward(preds)
            return [-ypred.value()  for ypred in preds]
        
    def predict_sentence(self,sentence,unk_flag=True,surprisal=True):
        """
        Outputs a sentence together with its predicted transitions probs as a pandas data frame
        @param sententce : a list of strings (words)
        @param unk_flag  : flags if this word is UNK TOKEN
        @param surprisals: also outputs -log2(p)
        @return a pandas DataFrame
        """
        X = []
        Y = []
        tokens = [NNLanguageModel.IOS_TOKEN,NNLanguageModel.IOS_TOKEN,NNLanguageModel.IOS_TOKEN]+sentence
        for (w3,w2,w1,wy) in zip(tokens,tokens[1:],tokens[2:],tokens[3:]):
            x,y = self.make_representation([w3,w2,w1],wy)
            X.append(x)
            Y.append(y)
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

    def sample_sentence(self,cutoff=40):
        """
        Randomly samples a sentence from the conditional distribution.
        @param cutoff ; max size of generated strings
        @return : a list of words strings
        """
        ios = self.word_codes[NNLanguageModel.IOS_TOKEN]
        eos = self.word_codes[NNLanguageModel.EOS_TOKEN]
        sentence = [ios,ios,ios]

        tok = self.sample_token(sentence)
        while tok != eos and len(sentence) < cutoff+3:
            sentence.append(tok)
            tok = self.sample_token(sentence[-3:])
            
        return [self.rev_word_codes[tok] for tok in sentence][3:]

    def sample_token(self,sentence):
        """
        Samples a token from the conditional distrib
        @param sentence: a list of token indexes
        @return the index of the sampled token
        """
        ctxt = sentence[-3:]
        if self.tied:
            dy.renew_cg()
            W = dy.parameter(self.hidden_weights)
            E = dy.parameter(self.embedding_matrix)
            embeddings = [dy.pick(E, x) for x in ctxt]
            xdense     = dy.concatenate(embeddings)
            ypred     = dy.softmax(E * dy.tanh( W * xdense ))

            Ypred = np.array(ypred.value())
            Ypred /= Ypred.sum()  #fixes numerical instabilities
            return choice(self.lexicon_size,p=Ypred)
            
        else:
            dy.renew_cg()
            O = dy.parameter(self.output_weights)
            W = dy.parameter(self.hidden_weights)
            E = dy.parameter(self.embedding_matrix)
            for x,y in zip(X,Y):
                embeddings = [dy.pick(E, x) for x in ctxt]
                xdense     = dy.concatenate(embeddings)
                ypred     = dy.softmax(O * dy.tanh( W * xdense ),y)

            Ypred = np.array(ypred.value())
            Ypred /= Ypred.sum()   #fixes numerical instabilities
            return choice(self.lexicon_size,p=Ypred)

    def train_nn_lm(self,\
                    train_sentences,\
                    validation_sentences,\
                    lr=0.001,\
                    hidden_dropout=0.1,\
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
        print("max_epochs = %d\ntraining examples [N] = %d\nBatch size = %d\nDropout = %f\nlearning rate = %f"%(max_epochs,training_generator.N,batch_size,hidden_dropout,lr),flush=True)

        
        #(3) Model structure
        self.model = dy.ParameterCollection()
        self.hidden_weights   = self.model.add_parameters((self.hidden_size,self.embedding_size*self.input_length))
        if glove_file is None:
            self.embedding_matrix = self.model.add_parameters((self.lexicon_size,self.embedding_size))
        else:
            self.embedding_matrix = self.model.parameters_from_numpy(self.read_glove_embeddings(glove_file))
        if not self.tied:
            self.output_weights =  self.model.add_parameters((self.lexicon_size,self.hidden_size))
            
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
                    W = dy.parameter(self.hidden_weights)
                    E = dy.parameter(self.embedding_matrix)
                    batched_X        = zip(*X) #transposes the X matrix
                    lookups          = [dy.pick_batch(E,xcolumn) for xcolumn in batched_X]
                    xdense           = dy.concatenate(lookups)
                    ybatch_preds     = dy.pickneglogsoftmax_batch(E * dy.dropout(dy.tanh( W * xdense ),hidden_dropout),Y)
                    loss = dy.sum_batches(ybatch_preds)
                else:
                    dy.renew_cg()
                    O = dy.parameter(self.output_weights)
                    W = dy.parameter(self.hidden_weights)
                    E = dy.parameter(self.embedding_matrix)
                    batched_X        = zip(*X) #transposes the X matrix
                    lookups          = [dy.pick_batch(E,xcolumn) for xcolumn in batched_X]
                    xdense           = dy.concatenate(lookups)
                    ybatch_preds     = dy.pickneglogsoftmax_batch(O * dy.dropout(dy.tanh(W * xdense),hidden_dropout),Y)
                    loss = dy.sum_batches(ybatch_preds)

                N+=len(Y)
                L += loss.value()
                loss.backward()
                trainer.update()
                
            end_t = time.time()
            
            #validation and auto-saving
            Xvalid,Yvalid = validation_generator.batch_all()
            valid_nll = -sum(self.predict_logprobs(Xvalid,Yvalid))
            valid_ppl = exp(valid_nll/len(Yvalid))
            history_log.append((e,end_t-start_t,L,exp(L/N),valid_nll,valid_ppl))
            print('Epoch %d (%.2f sec.) NLL (train) = %f, PPL (train) = %f, NLL(valid) = %f, PPL(valid) = %f'%tuple(history_log[-1]),flush=True)

            if valid_nll == min(valid_nll,min_nll):
                min_nll = valid_nll
                self.save_model('best_model_dump',epoch=e)

        return pd.DataFrame(history_log,columns=['epoch','wall_time','NLL(train)','PPL(train)','NLL(dev)','PPL(dev)'])

    
    def eval_dataset(self,sentences):
        """
        @param sentences : a list of list of words
        @return : a couple (negative log likelihood,perplexity) 
        """
        data_generator = self.make_data_generator(sentences,len(sentences))
        X,Y            = data_generator.batch_all()
        nll            = -sum(self.predict_logprobs(X,Y))
        ppl            = exp(nll/len(Y))
        return (nll,ppl)
        
            
    @staticmethod
    def load_model(dirname):

        #TODO: forgot to store refs to embedding_matrix and weights
        
        istream = open(os.path.join(dirname,'params.pkl'),'rb')
        params = pickle.load(istream)
        istream.close()
        
        g = NNLanguageModel(input_length=params['input_length'],\
                            embedding_size=params['embedding_size'],\
                            hidden_size=params['hidden_size'],
                            tiedIO=params['tied'])

        g.lexicon_size    = params['lexicon_size']
                            
        istream = open(os.path.join(dirname,'words.pkl'),'rb')
        g.word_codes = pickle.load(istream)
        istream.close()
    
        g.rev_word_codes = ['']*len(g.word_codes)
        for w,idx  in g.word_codes.items():
            g.rev_word_codes[idx] = w

        g.model              = dy.ParameterCollection()
        g.hidden_weights     = g.model.add_parameters((g.hidden_size,g.embedding_size*g.input_length))
        g.embedding_matrix   = g.model.add_parameters((g.lexicon_size,g.embedding_size))
        if not g.tied:
            g.output_weights =  g.model.add_parameters((g.lexicon_size,g.hidden_size))
        g.model.populate(os.path.join(dirname,'model.prm'))
        return g

    
    def save_model(self,dirname,epoch= -1):
        """
        @param dirname:the name of a directory (existing or to create) where to save the model.
        @param epoch: if positive; stores the epoch at which this model was generated.
        """
        
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        #select parameters to save
        params = {'input_length':self.input_length,\
                  'lexicon_size':self.lexicon_size,\
                  'embedding_size':self.embedding_size,\
                  'hidden_size':self.hidden_size,\
                  'tied':self.tied}
        if epoch > 0:
            params['epoch'] = epoch
                  
        ostream = open(os.path.join(dirname,'params.pkl'),'wb')
        pickle.dump(params,ostream)
        ostream.close()

        ostream = open(os.path.join(dirname,'words.pkl'),'wb')
        pickle.dump(self.word_codes,ostream)
        ostream.close()
    
        self.model.save(os.path.join(dirname,'model.prm')) 

    @staticmethod
    def grid_search(train,\
                    dev,\
                    LR    = (0.01,0.001,0.0001),\
                    HSIZE = (100,200,400),\
                    ESIZE = (50,100,300),\
                    DPOUT = (0.1,0.2,0.3)):
        """
        Performs a grid search on hyperparameters and dumps whatever.
        This function should be called with care...
        @param train: the training treebank
        @param dev : the validation treebank
        @param LR: the learning rate for gradient descent
        @param HSIZE: size of the hidden layer
        @param ESIZE: size of the embeddings
        @param DPOUT: value of dropout on hiddden layer output
        """
        global_stats = []
        for lr in LR:
            for esize in ESIZE:
                for hsize in HSIZE:
                    for dpout in DPOUT:
                        lm = NNLanguageModel()
                        lm.hidden_size    = hsize
                        lm.embedding_size = esize
                        df = lm.train_nn_lm(train,\
                                            dev,\
                                            lr=lr,\
                                            batch_size=128,\
                                            hidden_dropout=dpout,\
                                            max_epochs=40,\
                                            glove_file='glove/glove.6B.%dd.txt'%(esize))
                        modstr = 'LM-lr=%f-esize=%d-hsize=%d-dpout=%f'%(lr,esize,hsize,dpout)
                        global_stats.append((modstr,df['loss'].iloc[-1],lm.perplexity(train),lm.perplexity(dev)))
                        print(global_stats[-1])
        #dumps summary at the end
        print('modname : train-loss : ppl-train : ppl-dev')                
        print('\n'.join(['%s : %f : %f : %f'%(modname,train_loss,pplt,ppld) for (modname,train_loss,pplt,ppld) in global_stats]))

        


if __name__ == '__main__':

    #read UD treebank
    #ttreebank = UDtreebank_reader('UD_English/en-ud-train.conllu')
    #dtreebank = UDtreebank_reader('UD_English/en-ud-dev.conllu')

    #read penn treebank
    ttreebank =  ptb_reader('ptb/ptb_train_50w.txt')
    dtreebank =  ptb_reader('ptb/ptb_valid.txt')
    
    #search for structure
    #NNLanguageModel.grid_search(ttreebank,dtreebank,LR=[0.001,0.0001],ESIZE=[300],HSIZE=[200,300],DPOUT=[0.1,0.2,0.3])
    #search for smoothing
    #NNLanguageModel.grid_search(ttreebank,dtreebank,LR=[0.001],HSIZE=[200],ESIZE=[300])    

    lm = NNLanguageModel(hidden_size=300,embedding_size=300,input_length=3,tiedIO=True)
    lm.train_nn_lm(ttreebank,dtreebank,lr=0.00001,hidden_dropout=0.4,batch_size=512,max_epochs=35,glove_file='glove/glove.6B.300d.txt')
    lm.save_model('final_model')

    test_treebank =  ptb_reader('ptb/ptb_test.txt')
    print(lm.eval_dataset(test_treebank))
    
    #lm2 =  NNLanguageModel.load_model('final_model')
    #for s in ttreebank[:5]:
    #    print(lm2.predict_sentence(s))

    #for _ in range(10):
    #    print(' '.join(lm.sample_sentence()))
