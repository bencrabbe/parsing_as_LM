import dynet as dy
from constree      import *
from lexicons      import *
from proc_monitors import *
from rnng_params   import *
from char_rnn      import *
from math          import exp
from numpy.random import rand

class RNNLM:

        START_TOKEN   = '<start>'
        UNKNOWN_TOKEN = '<unk>'
    
        def __init__(self,brown_clusters,word_embedding_size=250,hidden_size=300,char_embedding_size=50,vocab_thresh=1):
            
            self.word_embedding_size = word_embedding_size
            self.hidden_size         = hidden_size
            self.char_embedding_size = char_embedding_size
            self.brown_file          = brown_clusters
            self.vocab_thresh        = vocab_thresh
            self.dropout             = 0.0
            
        def allocate_params(self):
                """
                Allocates memory for the model parameters.
                """
                self.model                     = dy.ParameterCollection()
                self.word_embeddings           = self.model.add_lookup_parameters((self.lexicon.size(),self.word_embedding_size)) 
                self.rnn                       = dy.LSTMBuilder(2,self.word_embedding_size+self.char_embedding_size,self.hidden_size,self.model)          
                self.char_rnn                  = CharRNNBuilder(self.char_embedding_size,self.char_embedding_size,self.charset,self.model)
                self.word_softmax              = dy.ClassFactoredSoftmaxBuilder(self.hidden_size,self.brown_file,self.lexicon.words2i,self.model,bias=True)    
        
        def code_lexicons(self,treebank):
            known_vocabulary = []
            charset          = set([ ])
            for tree in treebank:
                tokens = tree.tokens()
                for word in tokens:
                    charset.update(list(word))
                known_vocabulary.extend(tokens)
            known_vocabulary = get_known_vocabulary(known_vocabulary,vocab_threshold=1)
            known_vocabulary.add(RNNLM.START_TOKEN)
            self.brown_file  = normalize_brown_file(self.brown_file,known_vocabulary,self.brown_file+'.unk2',UNK_SYMBOL=RNNLM.UNKNOWN_TOKEN)
            self.lexicon     = SymbolLexicon( list(known_vocabulary),unk_word=RNNLM.UNKNOWN_TOKEN)
            self.charset     = SymbolLexicon(list(charset))
            return self.lexicon

        def predict_logprobs(self,X):
            """
            Predicts log probabilities for a sentence X (list of words)
            Returns the NLL for this sentence.
            """
            Y  = X
            X  = [RNNLM.START_TOKEN] + X
            X.pop()             
            dy.renew_cg()
            
            state       = self.rnn.initial_state()
            xcodes      = [self.lexicon.index(x) for x in X]
            cembeddings = [self.char_rnn(x) for x in X]      #char embeddings
            lookups     = [dy.concatenate([self.word_embeddings[xidx],charE]) for (xidx,charE) in zip(xcodes,cembeddings)]
            outputs     = state.transduce(lookups)

            ycodes      = [self.lexicon.index(y) for y in Y]
            ypreds      = [self.word_softmax.neg_log_softmax(o,y) for (o,y) in zip(outputs,ycodes)]
            nll         = dy.esum(ypreds).value()
            return nll

        def eval_dataset(self,treebank_file,strip_trees=True):
            """
            Evaluates the model on a dataset and returns nll and perplexity
            """
            nll = 0
            N   = 0
            treebank = open(treebank_file)
            for line in treebank:
                if strip_trees: #sent is a tree
                    tree    = ConsTree.read_tree(line)
                    tokens  = tree.tokens() 
                else:
                    tokens  = line.split() 
                nll += self.predict_logprobs(tokens)
                N   += len(tokens)
            treebank.close()
            return nll,exp(nll/N)
    
        def read_glove_embeddings(self,glove_filename):
                """
                Reads embeddings from a glove filename and returns an embedding
                matrix for the parser vocabulary.
                @param glove_filename: the file where to read embeddings from
                @return an embedding matrix that can initialize an Embedding layer
                """
                print('Reading embeddings from %s ...'%glove_filename)
                #self.word_embeddings = self.model.add_lookup_parameters((self.lexicon.size(),self.word_embedding_size), init='glorot')                 
                istream = open(glove_filename)
                for line in istream:
                        values = line.split()
                        word = values[0]
                        widx = self.lexicon.index(word)
                        
                        if widx != self.lexicon.unk_index():
                                coefs = np.asarray(values[1:], dtype='float32')
                                self.word_embeddings.init_row(widx,coefs)

                istream.close()
                print('done.')
        
        def train_rnnlm(self,train_file,\
                             dev_file, \
                             lr=0.1, \
                             dropout=0.5,\
                             max_epochs=100):
                              
            #Trees preprocessing
            train_stream   = open(train_file)
            train_treebank = [ ] 

            
            for idx,line in enumerate(train_stream):
                t = ConsTree.read_tree(line)
                train_treebank.append(t)
            train_stream.close()
                 
            self.dropout = dropout
            self.code_lexicons(train_treebank)
            self.allocate_params()
            #external word embeddings
            self.read_glove_embeddings('glove.6B.300d.txt')
            trainer = dy.SimpleSGDTrainer(self.model,learning_rate=lr)
            min_ppl = float('inf') 
            for e in range(max_epochs): 
                nll        = 0
                N          = 0 
                for sent in train_treebank:
                    dy.renew_cg()
                    Y           = sent.tokens()   
                    X           = [RNNLM.START_TOKEN] + Y
                    X.pop()
                    state       = self.rnn.initial_state()
                    xcodes      = [self.lexicon.index(x) for x in X]
                    
                    cembeddings = [self.char_rnn(x) for x in X]      #char embeddings
                    lookups     = [dy.concatenate([self.word_embeddings[xidx],charE]) for (xidx,charE) in zip(xcodes,cembeddings)]
                    outputs     = state.transduce(lookups)

                    ycodes      = [self.lexicon.index(y) for y in Y]
                    losses      = [self.word_softmax.neg_log_softmax(dy.rectify(dy.dropout(o,self.dropout)),y) for (o,y) in zip(outputs,ycodes)]
                    
                    loss        = dy.esum(losses)
                    loss.backward() 
                    trainer.update()

                    nll        += loss.value()
                    N          += len(Y)
                    
                train_ppl       = exp(nll/N)
                dev_nll,dev_ppl = self.eval_dataset(dev_file)
                
                print("epoch",e,'train PPL:',train_ppl,'dev PPL',dev_ppl,flush=True)
                if dev_ppl <= min_ppl:
                    self.model.save('rnnlm_model.prm')
                    print('   >model saved<')
                    min_ppl = dev_ppl
                    
            self.model   = self.model.populate('rnnlm_model.prm')
            self.dropout = 0.0 
             
lm = RNNLM('ptb-250.brown',word_embedding_size=300,hidden_size=300)  
lm.train_rnnlm('ptb_train.mrg','ptb_dev.mrg',max_epochs=20,lr=0.1,dropout=0.6)
print('WSJ PPL',lm.eval_dataset('ptb_test.mrg')[1])
print('Prince PPL',lm.eval_dataset('prince/prince.en.txt',strip_trees=False)[1])
