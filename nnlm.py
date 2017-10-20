import os
import os.path
import pickle
import numpy as np
import pandas as pd


from math import log2,exp
from numpy.random import choice,rand
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Embedding,Flatten, Dropout
from keras_extra import EmbeddingTranspose,NumericDataGenerator
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
            
class DependencyTree:

    def __init__(self,tokens=None, edges=None):
        self.edges  = [] if edges is None else edges   #couples (gov_idx,dep_idx)
        self.tokens = ['#ROOT#'] if tokens is None else tokens #list of wordforms
    
    def __str__(self):
        gdict = dict([(d,g) for (g,d) in self.edges])
        return '\n'.join(['\t'.join([str(idx+1),tok,str(gdict[idx+1])]) for idx,tok in enumerate(self.tokens[1:])])

    def is_projective(self,root=0,ideps=None):
        """
        Tests if a dependency tree is projective.
        @param root : the index of the root node
        @param ideps: a dict index -> list of immediate dependants.
        @return: (a boolean stating if the root is projective , a list
        of children idxes)
        """
        if ideps is None:#builds dict if not existing
            ideps = {}
            for gov,dep in self.edges:
                if gov in ideps:
                    ideps[gov].append(dep)
                else:
                    ideps[gov] = [dep]
        
        allc = [root]                              #reflexive
        if root not in ideps:                      #we have a leaf
            return (True,allc)
        for child in ideps[root]:
            proj,children = self.is_projective(child,ideps)
            if not proj:
                return (proj,None)
            allc.extend(children)                   #transitivity

        allc.sort()                                 #checks projectivity
        for _prev,_next in zip(allc,allc[1:]):
            if _next != _prev+1:
                return (False,None)
        return (True, allc)
    
    @staticmethod
    def read_tree(istream):
        """
        Reads a tree from an input stream in CONLL-U format.
        Currently ignores empty nodes and compound spans.
        
        @param istream: the stream where to read from
        @return: a DependencyTree instance 
        """
        deptree = DependencyTree()
        bfr = istream.readline()
        while bfr:
            if bfr[0] == '#':
                bfr = istream.readline()
            elif (bfr.isspace() or bfr == ''):
                if deptree.N() > 0:
                    return deptree
                else:
                    bfr = istream.readline()
            else:
                line_fields = bfr.split()
                idx, word, governor_idx = line_fields[0],line_fields[1],line_fields[6]
                if not '.' in idx: #empty nodes have a dot (and are discarded here)
                    deptree.tokens.append(word)
                    deptree.edges.append((int(governor_idx),int(idx)))
                bfr = istream.readline()
        return None

    
    def accurracy(self,other):
        """
        Compares this dep tree with another by computing their UAS.
        Assumes this tree is the reference tree
        @param other: other dep tree
        @return : the UAS as a float
        """
        S1 = set(self.edges)
        S2 = set(other.edges)
        return len(S1.intersection(S2)) / len(S1)
    
    def N(self):
        """
        Returns the length of the input
        """
        return len(self.tokens)
    
    def __getitem__(self,idx):
        """
        Returns the token at index idx
        """
        return self.tokens[idx]
        

class NNLanguageModel:
    """
    A simple NNLM a la Bengio 2002.
    """
    #Undefined and unknown symbols
    UNDEF_TOKEN   = "__UNDEF__"
    UNKNOWN_TOKEN = "<unk>"
    EOS_TOKEN     = "__EOS__"
    
    def __init__(self):
        self.model            = None
        self.input_size       = 3    #num symbols fed to the network for predictions
        self.embedding_size   = 50
        self.hidden_size      = 100
        self.word_codes       = None  #TBD at train time or at param loading
        self.rev_word_codes   = None  #TBD at train time or at param loading
        self.lexicon_size     = 0     #TBD at train time or at param loading
        
    def __str__(self):
        s = ['INPUT SIZE : %d'%(self.input_size),\
            'EMBEDDING_SIZE : %d'%(self.embedding_size),\
            'HIDDEN_LAYER_SIZE : %d'%(self.hidden_size),\
            'LEXICON_SIZE : %d'%(self.lexicon_size)]
        return '\n'.join(s)

    def read_glove_embeddings(self,glove_filename):
        """
        Reads embeddings from a glove filename and returns an embedding
        matrix for the parser vocabulary.
        @param glove_filename: the file where to read embeddings from
        @return an embedding matrix that can initialize an Embedding layer
        """
        print('Reading embeddings from %s ...'%glove_filename)

        embedding_matrix = (rand(self.lexicon_size,self.embedding_size) - 0.5)/10.0 #like a Keras uniform initializer 

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
        for Keras.
        @param xcontext: the preceding words
        @return a couple (X,Y) or just X if no prediction is given as param
        """
        Nx   = len(xcontext)
        X    = [None]*self.input_size
        unk_code = self.word_codes[NNLanguageModel.UNKNOWN_TOKEN]
        X[0] = self.word_codes.get(xcontext[-1],unk_code) if  Nx > 0 else self.word_codes[NNLanguageModel.UNDEF_TOKEN]  
        X[1] = self.word_codes.get(xcontext[-2],unk_code) if  Nx > 1 else self.word_codes[NNLanguageModel.UNDEF_TOKEN]  
        X[2] = self.word_codes.get(xcontext[-3],unk_code) if  Nx > 2 else self.word_codes[NNLanguageModel.UNDEF_TOKEN]  
    
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
        lexicon = set([NNLanguageModel.UNKNOWN_TOKEN,NNLanguageModel.UNDEF_TOKEN,NNLanguageModel.EOS_TOKEN])
        for sentence in treebank:
            lexicon.update(sentence)
        self.lexicon_size = len(lexicon)
        self.rev_word_codes = list(lexicon)
        self.word_codes = dict([(s,idx) for (idx,s) in enumerate(self.rev_word_codes)])

    def make_data_generator(self,treebank,batch_size):
        """
        This returns a data generator suitable for use with Keras
        @param treebank: the treebank to encode
        @param batch_size: the size of the batches yielded by the generator
        """
        Y    = []
        X    = []
        for line in treebank:
            tokens = [NNLanguageModel.UNKNOWN_TOKEN,NNLanguageModel.UNKNOWN_TOKEN,NNLanguageModel.UNKNOWN_TOKEN]+line+[NNLanguageModel.EOS_TOKEN]
            for (w3,w2,w1,wy) in zip(tokens,tokens[1:],tokens[2:],tokens[3:]):
                x,y    = self.make_representation([w3,w2,w1],wy)
                X.append(x)
                Y.append(y)
                
        return NumericDataGenerator(X,Y,batch_size,self.lexicon_size,self.word_codes[NNLanguageModel.UNKNOWN_TOKEN])


    
    def predict(self,sentence,yvalue=None):
        """
        Returns the prediction of this model given configuration.
        if yvalue is a word string, then return the probability of this word
        if yvalue is None, then samples a prediction from the X|Y conditional distribution
        """
        X = np.array([self.make_representation(sentence,None)])
        Y = self.model.predict(X,batch_size=1)[0]

        if yvalue is None:
            return self.rev_word_codes[choice(self.lexicon_size,p=Y)] 
        else:
            return Y[self.word_codes.get(yvalue,self.word_codes[NNLanguageModel.UNKNOWN_TOKEN])]


    def predict_sentence(self,sentence):
        """
        Outputs a corpus together with its transitions probs as a data frame
        @param sent_list: a list of list of strings
        """
        X = []
        Y = []
        tokens = [NNLanguageModel.UNKNOWN_TOKEN,NNLanguageModel.UNKNOWN_TOKEN,NNLanguageModel.UNKNOWN_TOKEN]+sentence
        for (w3,w2,w1,wy) in zip(tokens,tokens[1:],tokens[2:],tokens[3:]):
            x,y = self.make_representation([w3,w2,w1],wy)
            X.append(x)
            Y.append(y)
        preds = self.model.predict(np.array(X))
        records = []
        for idx,yref in enumerate(Y):
            records.append( ( sentence[idx], sentence[idx] not in self.word_codes, preds[idx,yref],log2(preds[idx,yref])))
        return pd.DataFrame(records,columns=['token','unk_word','cond_prob','cond_log2prob'])
    
            
    def perplexity(self,treebank,uniform=False):
        """
        Computes the perplexity of an LM on a treebank
        @param uniform : outputs a perplexity that would be produced
        by an uniform distribution.
        """
        cross_entropy = 0
        N             = 0

        uniform_prob  = 1/len(self.word_codes)
        for sentence in treebank:
            X = []
            Y = []
            tokens = [NNLanguageModel.UNDEF_TOKEN,NNLanguageModel.UNDEF_TOKEN,NNLanguageModel.UNDEF_TOKEN]+sentence+[NNLanguageModel.EOS_TOKEN]
            for (w3,w2,w1,y) in zip(tokens,tokens[1:],tokens[2:],tokens[3:]):
                x,y = self.make_representation([w3,w2,w1],y)
                X.append(x)
                Y.append(y)
            if uniform:
                 cross_entropy += sum(log2(uniform_prob) for _ in range(len(Y)))
            else:
                preds = self.model.predict(X)
                cross_entropy += sum( log2(preds[idx,yref]+np.finfo(float).eps) for idx,yref in enumerate(Y) )
            N += len(Y)
        return 2**(-cross_entropy/N)
    
    def sample_sentence(self):

        sentence = [NNLanguageModel.UNDEF_TOKEN,NNLanguageModel.UNDEF_TOKEN,NNLanguageModel.UNDEF_TOKEN]
        action = self.predict(sentence)
        while True:
            if action == NNLanguageModel.EOS_TOKEN:
                 if len(sentence) > 20:
                    break
            else:
                sentence.append(action)
            action = self.predict(sentence)
        return sentence

    def train_nn_lm(self,\
                    treebank_train,\
                    treebank_validation,\
                    lr=0.001,\
                    hidden_dropout=0.1,\
                    batch_size=100,\
                    max_epochs=100,\
                    alpha_lex=-1,\
                    glove_file=None):
                      
        """
        Locally trains a model with a static oracle and a standard feedforward NN.  
        @param treebank : a list of dependency trees
        """
        #(1) build dictionaries
        self.code_symbols(treebank_train) 
        print("Dictionaries built.")

        #(2) read off treebank and build keras data set
        print("Encoding dataset from %d sentences."%len(treebank_train))
        training_generator = self.make_data_generator(treebank_train,batch_size)
        validation_generator = self.make_data_generator(treebank_validation,batch_size)
        if alpha_lex < 0:
            alpha_lex = training_generator.guess_alpha(validation_generator.oov_rate())

        print(self)
        print("training examples [N] = %d\nBatch size = %d\nDropout = %f\nlearning rate = %f\nalpha-lex = %f"%(training_generator.N,batch_size,hidden_dropout,lr,alpha_lex))

        #(3) Model structure
        self.model = Sequential()
        input_embedding = None
        
        if glove_file == None:
            input_embedding = Embedding(self.lexicon_size,self.embedding_size,input_length=self.input_size)
            self.model.add( input_embedding )
        else:
            input_embedding = Embedding(self.lexicon_size,\
                                        self.embedding_size,\
                                        input_length=self.input_size,\
                                        trainable=True,\
                                        weights=[self.read_glove_embeddings(glove_file)]) 
            self.model.add(input_embedding)
    
        self.model.add(Flatten())                   #concatenates the embeddings layers
        self.model.add(Dense(self.hidden_size))
        self.model.add(Activation('tanh'))
        if hidden_dropout > 0:
            self.model.add(Dropout(hidden_dropout))
        #self.model.add(Dense(self.lexicon_size))
        self.model.add(EmbeddingTranspose(self.lexicon_size,input_embedding))
        self.model.add(Activation('softmax'))
        sgd = Adam(lr=lr,beta_1=0.0)
        self.model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])         #also works fine with adam
        #(4) Fitting
        lr_scheduler = ReduceLROnPlateau( monitor='val_loss',factor=0.1,patience=5,verbose=1)
        checkpoint = ModelCheckpoint('temporary-model-{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        ntrain_batches = max(1,round(training_generator.N/batch_size))
        nvalidation_batches = max(1,round(validation_generator.N/batch_size))
        log = self.model.fit_generator(generator = training_generator.generate(alpha=alpha_lex),\
                                       validation_data = validation_generator.generate(),\
                                       epochs=max_epochs,\
                                       steps_per_epoch = ntrain_batches,\
                                       validation_steps = nvalidation_batches,\
                                       callbacks = [checkpoint,lr_scheduler])
        return pd.DataFrame(log.history)

    @staticmethod
    def load_model(dirname):

        g = NNLanguageModel()
         
        istream = open(os.path.join(dirname,'params.pkl'),'rb')
        params = pickle.load(istream)
        istream.close()
        
        g.lexicon_size = params['lexicon_size']
        g.input_size  = params['input_size']

        istream = open(os.path.join(dirname,'words.pkl'),'rb')
        g.word_codes = pickle.load(istream)
        istream.close()
    
        g.rev_word_codes = ['']*len(g.word_codes)
        for w,idx  in g.word_codes.items():
            g.rev_word_codes[idx] = w

        g.model = load_model(os.path.join(dirname,'model.prm'))
        return g
    
    def save_model(self,dirname):
        
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        #select parameters to save
        params = {'input_size':self.input_size,'lexicon_size':self.lexicon_size}

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

def UDtreebank_reader(filename):
    treebank = []
    istream = open(filename)
    dtree = DependencyTree.read_tree(istream)
    while dtree != None:
        if dtree.is_projective():
            treebank.append(dtree.tokens)
        dtree = DependencyTree.read_tree(istream)
    istream.close()
    return treebank

def ptb_reader(filename):
    istream = open(filename)
    treebank = []
    for line in istream:
        treebank.append(line.split())
    istream.close()
    return treebank

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

    lm = NNLanguageModel()
    lm.hidden_size    = 300
    lm.embedding_size = 300
    lm.train_nn_lm(ttreebank,dtreebank,lr=0.00001,alpha_lex=0,hidden_dropout=0.3,batch_size=128,max_epochs=240,\
                    glove_file='glove/glove.6B.300d.txt')
    lm.save_model('testLM')
    #lm = NNLanguageModel.load_model('testLM')
    #print('PPL-T = ',lm.perplexity(ttreebank),'PPL-D = ',lm.perplexity(dtreebank),'PPL-D(control) = ',lm.perplexity(dtreebank,uniform=True))
    #for sentence in dtreebank[:10]:
    #    df = lm.predict_sentence(sentence)
    #    print(df)
    
    #lm = NNLanguageModel.load_model('testLM')
    #for _ in range(10):
    #    print(' '.join(lm.sample_sentence()[3:]))
