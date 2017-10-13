#! /usr/bin/env python

#DATA REPRESENTATION
#TODO : Exclude non projective structures ? (almost done)
#TODO : termination is crap, maybe put the dummy root at the end...
#TODO : what to do with pos tags ?

import os
import os.path
import pickle
import numpy as np
import pandas as pd

from math import log,exp
from random import shuffle
from numpy.random import choice,rand
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Embedding,Flatten, Dropout
from keras.optimizers import RMSprop

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

class StackNode(object):
    """
    This is a class for storing structured nodes in the parser's stack 
    """
    __slots__ = ['root','ilc','irc','starlc','rc_node']

    def __init__(self,root_idx):
        self.root   = root_idx
        self.ilc    = root_idx
        self.irc    = root_idx
        self.starlc = root_idx
        self.rc_node = None
                        
    def copy(self):
        other = StackNode(self.root)
        other.ilc     = self.ilc
        other.irc     = self.irc
        other.starlc  = self.starlc
        other.rc_node = self.rc_node
        return other
            
    def starrc(self):
        """
        Computes and returns the index of the righmost child of
        this node (reflexive transitive closure of RC(x))
        """
        if self.rc_node is None:
            return self.root
        else:
            return self.rc_node.starrc()

    def copy_left_arc(self):
        """
        Creates a copy of this node for when it governs a left arc
        """
        other = self.copy()
        other.starlc = min(self.starlc,other.starlc)
        other.ilc    = min(self.ilc,other.ilc) 
        return other
    
    def copy_right_arc(self,right_node):
        """
        Creates a copy of this node for when it governs a right arc
        @param right_node: the governed right StackNode.
        """
        other        = self.copy()
        other.irc     = max(self.irc,other.root)
        other.rc_node = right_node
        return other
        
    def __str__(self):
        return str(self.root)


class NumericDataGenerator:
    """
    Stores an encoded treebank and provides it with a python generator interface
    """
    #TODO : add validation set + unk word sampling for train 
    def __init__(self,X,Y,batch_size,nclasses):
        """
        @param X,Y the encoded X,Y values as lists
        @param batch_size:size of generated data batches
        @param nclasses: number of Y classes
        """
        assert(len(X)==len(Y))
        self.X,self.Y = np.array(X),Y
        self.nclasses = nclasses
        self.N = len(Y)
        self.idxes = list(range(self.N))

        self.batch_size = batch_size
        self.start_idx = 0
        
    def select_indexes(self):
        end_idx = self.start_idx+self.batch_size
        if end_idx >= self.N:
            shuffle(self.idxes)
            self.start_idx = 0
            end_idx = self.batch_size

        sidx = self.start_idx
        self.start_idx = end_idx
        return (sidx,end_idx)
            
    def generate(self):
        """
        The generator called by the fitting function.
        """
        while True:
            start_idx,end_idx = self.select_indexes()
            Y = np.zeros((self.batch_size,self.nclasses))
            yvals = self.Y[start_idx:end_idx]
            X     = self.X[start_idx:end_idx]
            for i,y in enumerate(yvals):
                Y[i,y] = 1.0
            yield (X,Y)
    
class ArcEagerGenerativeParser:

    #actions
    LEFTARC  = "L"
    RIGHTARC = "R"
    GENERATE = "G"
    PUSH     = "P"
    REDUCE   = "RD"
    
    #end of sentence (TERMINATE ACTION)
    TERMINATE = "E"

    #Undefined and unknown symbols
    UNDEF_TOKEN   = "__UNDEF__"
    UNKNOWN_TOKEN = "__UNK__"

    def __init__(self):
        #sets default generic params
        self.model            = None
        self.stack_size       = 3
        self.node_size        = 5                                #number of x-values per stack node
        self.input_size       = 18                               #num symbols fed to the network for predictions
        self.embedding_size   = 100
        self.hidden_size      = 1200
        self.word_codes       = None  #TBD at train time or at param loading
        self.actions_codes    = None  #TBD at train time or at param loading
        self.rev_action_codes = None #TBD at train time or at param loading
        self.actions_size     = 0  #TBD at train time or at param loading
        self.lexicon_size     = 0  #TBD at train time or at param loading
        
    def __str__(self):
        s = ['INPUT SIZE : %d'%(self.input_size),\
            'EMBEDDING_SIZE : %d'%(self.embedding_size),\
            'HIDDEN_LAYER_SIZE : %d'%(self.hidden_size),\
            'ACTIONS_SIZE : %d'%(self.actions_size),\
            'LEXICON_SIZE : %d'%(self.lexicon_size)]
        return '\n'.join(s)
            
    #TRANSITION SYSTEM
    def init_configuration(self,tokens):
        """
        Generates the init configuration 
        """
        #init config: S, None, 1... n, empty arcs, score=0
        return ([StackNode(0)],None,tuple(range(1,len(tokens))),[],0.0)
        
    def push(self,configuration,local_score=0.0):
        """
        Performs the push action and returns a new configuration
        """
        S,F,B,A,prefix_score = configuration
        return (S + [F], None,B,A,prefix_score+local_score) 

    def leftarc(self,configuration,local_score=0.0):
        """
        Performs the left arc action and returns a new configuration
        """
        S,F,B,A,prefix_score = configuration
        i,j = S[-1].root,F.root
        return (S[:-1],F.copy_left_arc(),B,A + [(j,i)],prefix_score+local_score) 

    def rightarc(self,configuration,local_score=0.0):
        S,F,B,A,prefix_score = configuration
        i,j = S[-1].root,F.root
        S[-1] = S[-1].copy_right_arc(F)
        return (S+[F],None, B, A + [(i,j)],prefix_score+local_score) 

    def reduce_config(self,configuration,local_score=0.0):
        S,F,B,A,prefix_score = configuration
        return (S[:-1],F,B,A,prefix_score+local_score)
    
    def generate(self,configuration,local_score=0.0):
        """
        Pseudo-Generates the next word (for parsing)
        """
        S,F,B,A,prefix_score = configuration
        return (S,StackNode(B[0]),B[1:],A,prefix_score+local_score) 

    def static_oracle(self,configuration,dtree):
        """
        A default static oracle.
        @param configuration: a parser configuration
        @param dtree: a dependency tree object
        @return the action to execute given config and reference arcs
        """
        S,F,B,A,score  = configuration
        reference_arcs = dtree.edges
        all_words   = range(dtree.N())

        if F is None and B:
            return (ArcEagerGenerativeParser.GENERATE,dtree.tokens[B[0]])     
        if S and F:
            i,j = S[-1].root, F.root
            if (j,i) in reference_arcs:
                return ArcEagerGenerativeParser.LEFTARC
            if (i,j) in reference_arcs:
                return ArcEagerGenerativeParser.RIGHTARC
            
        if S and any([(k,S[-1].root) in A for k in all_words]) \
             and all ([(S[-1].root,k) in A for k in all_words if (S[-1].root,k) in reference_arcs]):
                return ArcEagerGenerativeParser.REDUCE
        if not F is None:
            return ArcEagerGenerativeParser.PUSH
        return ArcEagerGenerativeParser.TERMINATE
        
    def static_oracle_derivation(self,dtree):
        """
        This generates a static oracle reference derivation from a sentence
        @param ref_parse: a DependencyTree object
        @return : the oracle derivation as a list of (Configuration,action) triples
        """
        sentence = dtree.tokens
        
        C = self.init_configuration(sentence)
        action = self.static_oracle(C,dtree)
        derivation = []
        
        while action != ArcEagerGenerativeParser.TERMINATE :
            derivation.append((C,action,sentence))
            if   action ==  ArcEagerGenerativeParser.PUSH:
                C = self.push(C)
            elif action == ArcEagerGenerativeParser.LEFTARC:
                C = self.leftarc(C)
            elif action == ArcEagerGenerativeParser.RIGHTARC:
                C = self.rightarc(C)
            elif action == ArcEagerGenerativeParser.REDUCE:
                C = self.reduce_config(C)
            else:
                action, w = action
                assert(action ==  ArcEagerGenerativeParser.GENERATE)
                C = self.generate(C)
            action = self.static_oracle(C,dtree)
            
        derivation.append((C,action,sentence))
        return derivation

    
    #CODING & SCORING SYSTEM
    def score(self,configuration,sentence,next_word=None,logp=False):
        """
        Scores all actions and returns the first legal action.
        """
        X = self.make_representation(config,None,sentence)
        action = self.model.predict(X,batch_size=1)
        

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

    def pprint_configuration(self,config,sentence,verbose=False):
        """
        Pretty prints a configuration
        """
        S,F,B,A,score=config

        stack = ''
        if len(S) > 3:
            stack = '[...] '
        if len(S) > 2:
            stack += ' '+sentence[S[-2].root] if not verbose else '(%s %s %s)' %(sentence[S[-2].root],sentence[S[-2].ilc],sentence[S[-2].irc])
        if len(S) > 1:
            stack += ' '+sentence[S[-1].root] if not verbose else '(%s %s %s)' %(sentence[S[-1].root],sentence[S[-1].ilc],sentence[S[-1].irc])
        focus = '_'
        if F is not None:
            focus = sentence[F.root] if not verbose else '(%s %s %s)' %(sentence[F.root],sentence[F.ilc],sentence[F.irc])

        return '(%s,%s,_,_)'%(stack,focus)
        
    def make_representation(self,config,action,sentence):
        """
        Turns a configuration into a couple of vectors (X,Y) data and
        outputs the coded configuration as a tuple of vectors suitable
        for Keras.
        @param configuration: a parser configuration
        @param action: the action code or None if the action is not known
        @param sentence: a list of tokens (strings)
        @return a couple (X,Y) or just X if no action is given as param
        """        
        S,F,B,A,score = config
        X  = [None] * self.input_size
        Ns = len(S)
        
        X[0] = self.word_codes[sentence[F.root]]      if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[1] = self.word_codes[sentence[F.ilc]]       if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[2] = self.word_codes[sentence[F.irc]]       if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[3] = self.word_codes[sentence[F.starlc]]    if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[4] = self.word_codes[sentence[F.starrc()]]  if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]

        X[5] = self.word_codes[sentence[S[-1].root]]     if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[6] = self.word_codes[sentence[S[-1].ilc]]      if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[7] = self.word_codes[sentence[S[-1].irc]]      if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[8] = self.word_codes[sentence[S[-1].starlc]]   if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[9] = self.word_codes[sentence[S[-1].starrc()]] if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]

        X[10] = self.word_codes[sentence[S[-2].root]]     if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[11] = self.word_codes[sentence[S[-2].ilc]]      if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[12] = self.word_codes[sentence[S[-2].irc]]      if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[13] = self.word_codes[sentence[S[-2].starlc]]   if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[14] = self.word_codes[sentence[S[-2].starrc()]] if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
    
        X[15] = self.word_codes[sentence[S[-3].root]]     if Ns > 2 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[16] = self.word_codes[sentence[S[-3].ilc]]      if Ns > 2 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[17] = self.word_codes[sentence[S[-3].irc]]      if Ns > 2 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        
        
        if action is None:
            return X
        else:
            Y = self.actions_codes[action]         
            return (X,Y)

    def code_symbols(self,treebank):
        """
        Codes lexicon (x-data) and the list of action (y-data)
        on integers.
        @param treebank: the treebank where to extract the data from
        """
        #lexical coding
        lexicon = set([ArcEagerGenerativeParser.UNKNOWN_TOKEN, ArcEagerGenerativeParser.UNDEF_TOKEN])
        for dtree in treebank:
            lexicon.update(dtree.tokens)
        self.lexicon_size = len(lexicon)
        self.word_codes = dict([(s,idx) for (idx,s) in enumerate(lexicon)])

        actions = [ArcEagerGenerativeParser.LEFTARC,\
                   ArcEagerGenerativeParser.RIGHTARC,\
                   ArcEagerGenerativeParser.PUSH,\
                   ArcEagerGenerativeParser.REDUCE,\
                   ArcEagerGenerativeParser.TERMINATE]
        actions.extend([(ArcEagerGenerativeParser.GENERATE,w)  for w in lexicon])
        self.actions_codes = dict([(s,idx) for (idx,s) in enumerate(actions)])
        self.rev_action_codes = actions
        self.actions_size  = len(actions) 
        
    def static_nn_train(self,treebank,lr=0.001,hidden_dropout=0.1,batch_size=100,max_epochs=100,glove_file=None):
        """
        Locally trains a model with a static oracle
        and a standard feedforward NN.  
        @param treebank : a list of dependency trees
        """
        #(1) build dictionaries
        self.code_symbols(treebank) 

        print("Dictionaries built.")
        print("Encoding dataset from %d trees."%len(treebank))
        
        #(2) read off treebank and build keras data set
        Y    = []
        X    = []
                
        for dtree in treebank:
            Deriv = self.static_oracle_derivation(dtree)
            for (config,action,sentence) in Deriv:
                x,y    = self.make_representation(config,action,sentence)
                X.append(x)
                Y.append(y)
                
        training_generator =  NumericDataGenerator(X,Y,batch_size,self.actions_size).generate()
                
        #(3) make network
        print(self)
        print("training examples [N] = %d\nBatch size = %d\nDropout = %f\nlearning rate = %f"%(len(Y),batch_size,hidden_dropout,lr))
        self.model = Sequential()
        
        if glove_file == None:
            self.model.add(Embedding(self.lexicon_size,self.embedding_size,input_length=self.input_size))
        else:
            self.model.add(Embedding(self.lexicon_size,\
                                     self.embedding_size,\
                                     input_length=self.input_size,\
                                     weights=[self.read_glove_embeddings(glove_file)]))

        self.model.add(Flatten())        #concatenates the embeddings layers
        self.model.add(Dense(self.hidden_size))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(hidden_dropout))
        self.model.add(Dense(self.actions_size))
        self.model.add(Activation('softmax'))
        rms = RMSprop(lr=lr)
        self.model.compile(optimizer=rms,loss='categorical_crossentropy',metrics=['accuracy'])
        nbatches = max(1,round(len(Y)/batch_size))
        log = self.model.fit_generator(generator = training_generator,epochs=max_epochs,steps_per_epoch = nbatches)
        return pd.DataFrame(log.history)

        
    #PERSISTENCE
    @staticmethod
    def load_parser(dirname):
        
        p = ArcEagerGenerativeParser()
                
        istream = open(os.path.join(dirname,'params.pkl'),'rb')
        params = pickle.load(istream)
        istream.close()
        
        p.stack_size = params['stack_size']
        p.node_size  = params['node_size']
        p.input_size = params['input_size']
        p.embedding_size = params['embedding_size']
        p.hidden_size = params['hidden_size']
        p.actions_size = params['actions_size']
        p.lexicon_size = params['lexicon_size']

        istream = open(os.path.join(dirname,'words.pkl'),'rb')
        p.word_codes = pickle.load(istream)
        istream.close()
    
        istream = open(os.path.join(dirname,'actions.pkl'),'rb')
        p.actions_codes = pickle.load(istream)
        istream.close()
        
        p.rev_action_codes = ['']*len(p.actions_codes)
        for A,idx  in p.actions_codes.items():
            p.rev_action_codes[idx] = A
        p.actions_size   = len(p.actions_codes)  
        p.lexicon_size   = len(p.word_codes)   

        p.model = load_model(os.path.join(dirname,'model.prm'))
        
        return p
        
    def save_parser(self,dirname):
        
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        #select parameters to save
        params = {'stack_size':self.stack_size,\
                  'node_size':self.node_size,\
                  'input_size':self.input_size,\
                  'embedding_size':self.embedding_size,\
                  'hidden_size':self.hidden_size,\
                  'actions_size':self.actions_size,\
                  'lexicon_size':self.lexicon_size}

        ostream = open(os.path.join(dirname,'params.pkl'),'wb')
        pickle.dump(params,ostream)
        ostream.close()

        ostream = open(os.path.join(dirname,'words.pkl'),'wb')
        pickle.dump(self.word_codes,ostream)
        ostream.close()
    
        ostream = open(os.path.join(dirname,'actions.pkl'),'wb')
        pickle.dump(self.actions_codes,ostream)
        ostream.close()
        
        self.model.save(os.path.join(dirname,'model.prm')) 
        

    #SELF-TESTS and eval
    def test_and_print(self,treebank):
        """
        Performs local tests and pretty prints cases of failure
        """
        for dtree in treebank:
            Deriv = self.static_oracle_derivation(dtree)
            for (config,action,sentence) in Deriv:
                self.predict_and_print(config,action,sentence)
                
    
    def predict_and_print(self,configuration,ref_action,sentence):
        """
        Compares the argmax prediction from configuration with ref_action.
        pretty prints incorrect cases.
        @return True if agreement, false otherwise
        """
        X = np.array([self.make_representation(configuration,None,sentence)])
        Y = self.model.predict(X,batch_size=1)[0]
        pred_action = self.rev_action_codes[np.argmax(Y)]
        pred_prob = Y[self.actions_codes[pred_action]]
        ref_prob  = Y[self.actions_codes[ref_action]]
        
        if pred_action != ref_action:
            S,F,B,A,score = configuration
            Ns = len(S)  
            s2 = sentence[S[-3].root] if Ns > 2 else  ArcEagerGenerativeParser.UNDEF_TOKEN
            s1 = sentence[S[-2].root] if Ns > 1 else  ArcEagerGenerativeParser.UNDEF_TOKEN
            s0 = sentence[S[-1].root] if Ns > 0 else  ArcEagerGenerativeParser.UNDEF_TOKEN
            fr = sentence[F.root]     if F is not None else ArcEagerGenerativeParser.UNDEF_TOKEN
            fl = sentence[F.ilc]      if F is not None else ArcEagerGenerativeParser.UNDEF_TOKEN
            frt = sentence[F.irc]      if F is not None else ArcEagerGenerativeParser.UNDEF_TOKEN
            print('ERROR:%s : %s with p=%10.9f is predicted as %s with p=%10.9f'%(self.pprint_configuration(configuration,sentence),ref_action,ref_prob,pred_action,pred_prob))
            return False
        else:
            S,F,B,A,score = configuration
            Ns = len(S)
            s2 = sentence[S[-3].root] if Ns > 2 else  ArcEagerGenerativeParser.UNDEF_TOKEN
            s1 = sentence[S[-2].root] if Ns > 1 else  ArcEagerGenerativeParser.UNDEF_TOKEN
            s0 = sentence[S[-1].root] if Ns > 0 else  ArcEagerGenerativeParser.UNDEF_TOKEN
            fr = sentence[F.root]     if F is not None else ArcEagerGenerativeParser.UNDEF_TOKEN
            fl = sentence[F.ilc]      if F is not None else ArcEagerGenerativeParser.UNDEF_TOKEN
            frt = sentence[F.irc]     if F is not None else ArcEagerGenerativeParser.UNDEF_TOKEN
            print('CORRECT:%s : %s with p=%10.9f is predicted correctly.'%(self.pprint_configuration(configuration,sentence),ref_action,ref_prob))
            return True
        
        
class ArcEagerGenerator:
    """
    This is a stochastic language generator for the language models built by
    this module. 
    """
    #actions
    LEFTARC  = "L"
    RIGHTARC = "R"
    GENERATE = "G"
    PUSH     = "P"
    REDUCE   = "RD"
    
    #end of sentence (TERMINATE ACTION)
    TERMINATE = "E"

    #Undefined and unknown symbols
    UNDEF_TOKEN   = "__UNDEF__"
    UNKNOWN_TOKEN = "__UNK__"

    def __init__(self):
        
        self.model             = None
        self.stack_size        = 3
        self.node_size         = 5
        self.input_size        = 18
        self.word_codes        = None  
        self.actions_codes     = None  #TBD at train time or at param loading
        self.rev_action_codes  = None #TBD at train time or at param loading
        self.actions_size      = 0  #TBD at train time or at param loading
        self.lexicon_size      = 0  #TBD

    #TRANSITION SYSTEM
    def init_configuration(self,local_score = 1.0):
        """
        Generates the init configuration 
        """
        #init config: S, None,empty terminals, empty arcs,score=0
        return ([StackNode(0)],None,('#ROOT#',),[],log(local_score))

    def push(self,configuration,local_score=0.0):
        """
        Performs the push action and returns a new configuration
        """
        S,F,T,A,prefix_score = configuration
        return (S + [F], None,T,A,prefix_score+log(local_score)) 

    def leftarc(self,configuration,local_score=0.0):
        """
        Performs the left arc action and returns a new configuration
        """
        S,F,B,A,prefix_score = configuration
        i,j = S[-1].root,F.root
        return (S[:-1],F.copy_left_arc(),B,A + [(j,i)],prefix_score+log(local_score)) 

    def rightarc(self,configuration,local_score=0.0):
        S,F,B,A,prefix_score = configuration
        i,j = S[-1].root,F.root
        S[-1] = S[-1].copy_right_arc(F)
        return (S+[F],None, B, A + [(i,j)],prefix_score+log(local_score)) 

    def reduce_config(self,configuration,local_score=0.0):
        S,F,B,A,prefix_score = configuration
        return (S[:-1],F,B,A,prefix_score+log(local_score))
    
    def generate(self,configuration,wordform,local_score=0.0):
        """
        Pseudo-Generates the next word (for parsing)
        """
        S,F,T,A,prefix_score = configuration
        return (S,StackNode(len(T)),T+(wordform,),A,prefix_score+log(local_score)) 
        
    def generate_sentence(self,max_len=2000,lex_stats=False):
        """
        @param lex_stats: generate a table with word,log(prefix_prob),log(local_prob),num_actions
        """
        C = self.init_configuration()
        derivation = [C]
        action,score =  self.stochastic_oracle(C,is_first_action=True)
        stats = []
        while action != ArcEagerGenerator.TERMINATE and len(derivation) < max_len:
            
            if action == ArcEagerGenerator.LEFTARC:
                C = self.leftarc(C,score)
            elif action == ArcEagerGenerator.RIGHTARC:
                C = self.rightarc(C,score)
            elif action == ArcEagerGenerator.REDUCE:
                C = self.reduce_config(C,score)
            elif action == ArcEagerGenerator.PUSH:
                C = self.push(C,score)
            else:
                action, w = action
                assert(action == ArcEagerGenerator.GENERATE)
                C = self.generate(C,w,score)
                stats.append((w,C[4],log(score),len(derivation)))
            derivation.append(C)

            action,score = self.stochastic_oracle(C)
            
        if lex_stats:
            df = pd.DataFrame(stats,columns=['word','log(P(deriv_prefix))','log(P(local))','nActions'])
            return df
        else:
            return derivation

    
    #CODING & SCORING
    def stochastic_oracle(self,configuration,is_first_action=False):
        S,F,terminals,A,score = configuration
        X = np.array([self.make_representation(configuration)])
        Y = self.model.predict(X,batch_size=1)[0]
        
        def has_governor(node_idx,arc_list):
            """
            Checks if a node has a governor
            @param node_idx: the index of the node
            @arc_list: an iterable over arc tuples
            @return a boolean
            """
            return any(node_idx  == didx for (gidx,didx) in arc_list)
        
        if is_first_action:#otherwise predicts terminate with p=1.0 because init config is also the config right before calling terminate
            while True:
                action_code = choice(self.actions_size)#uniform draw
                action_score = 1.0/self.actions_size
                action = self.rev_action_codes[action_code]
                if type(action) == tuple:#this is a generate action
                    return (action,action_score)            
        if not S or F is None or S[-1].root == 0 or not has_governor(S[-1].root,A):
            Y[self.actions_codes[ArcEagerGenerator.LEFTARC]] = 0.0
        if not S or F is None:
            Y[self.actions_codes[ArcEagerGenerator.RIGHTARC]] = 0.0
        if F is None or not S or not has_governor(S[-1].root,A):
            Y[self.actions_codes[ArcEagerGenerator.REDUCE]] = 0.0
        if F is None:
            Y[self.actions_codes[ArcEagerGenerator.PUSH]] = 0.0
        if F is not None:
            la = Y[self.actions_codes[ArcEagerGenerator.LEFTARC]]
            ra = Y[self.actions_codes[ArcEagerGenerator.RIGHTARC]]
            r  = Y[self.actions_codes[ArcEagerGenerator.REDUCE]]
            p  = Y[self.actions_codes[ArcEagerGenerator.PUSH]]
            t  = Y[self.actions_codes[ArcEagerGenerator.TERMINATE]]
            
            Y = np.zeros(self.actions_size)
            
            Y[self.actions_codes[ArcEagerGenerator.LEFTARC]]   = la
            Y[self.actions_codes[ArcEagerGenerator.RIGHTARC]]  = ra
            Y[self.actions_codes[ArcEagerGenerator.REDUCE]]    = r
            Y[self.actions_codes[ArcEagerGenerator.PUSH]]      = p
            Y[self.actions_codes[ArcEagerGenerator.TERMINATE]] = t
            
        Z = Y.sum()
        if Z == 0.0:#no action possible, trapped in dead-end, abort.
            return (ArcEagerGenerator.TERMINATE,np.finfo(float).eps)
            
        Y /= Z
        action_code = choice(self.actions_size,p=Y)
        action_score = Y[action_code]
        action = self.rev_action_codes[action_code]

        #print distribution:
        #print ('kbest')
        #kbest = sorted([(p,idx) for (idx,p) in enumerate(Y)],reverse=True)[:20]
        #for p,idx in kbest:
        #    print(idx,self.rev_action_codes[idx],p)
                
        return (action,action_score)


     
    def make_representation(self,config):
        """
        Turns a configuration into a vector of X  data and
        outputs a list of actions sorted by decreasing score.
        @param configuration: a parser configuration
        @return a list X of predictors.
        """        
        S,F,sentence,A,score = config
        X  = [None] * self.input_size
        Ns = len(S)
        
        X[0] = self.word_codes[sentence[F.root]]      if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[1] = self.word_codes[sentence[F.ilc]]       if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[2] = self.word_codes[sentence[F.irc]]       if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[3] = self.word_codes[sentence[F.starlc]]    if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[4] = self.word_codes[sentence[F.starrc()]]  if F is not None else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]

        X[5] = self.word_codes[sentence[S[-1].root]]     if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[6] = self.word_codes[sentence[S[-1].ilc]]      if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[7] = self.word_codes[sentence[S[-1].irc]]      if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[8] = self.word_codes[sentence[S[-1].starlc]]   if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[9] = self.word_codes[sentence[S[-1].starrc()]] if Ns > 0 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]

        X[10] = self.word_codes[sentence[S[-2].root]]     if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[11] = self.word_codes[sentence[S[-2].ilc]]      if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[12] = self.word_codes[sentence[S[-2].irc]]      if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[13] = self.word_codes[sentence[S[-2].starlc]]   if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[14] = self.word_codes[sentence[S[-2].starrc()]] if Ns > 1 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]

        X[15] = self.word_codes[sentence[S[-3].root]]     if Ns > 2 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[16] = self.word_codes[sentence[S[-3].ilc]]      if Ns > 2 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        X[17] = self.word_codes[sentence[S[-3].irc]]      if Ns > 2 else self.word_codes[ArcEagerGenerativeParser.UNDEF_TOKEN]
        
        return X

           
    @staticmethod
    def load_generator(dirname):
        
        g = ArcEagerGenerator()
                
        istream = open(os.path.join(dirname,'params.pkl'),'rb')
        params = pickle.load(istream)
        istream.close()
        
        g.stack_size = params['stack_size']
        g.node_size  = params['node_size']
        g.input_size = params['input_size']

        istream = open(os.path.join(dirname,'words.pkl'),'rb')
        g.word_codes = pickle.load(istream)
        istream.close()
    
        istream = open(os.path.join(dirname,'actions.pkl'),'rb')
        g.actions_codes = pickle.load(istream)
        istream.close()
        
        g.rev_action_codes = ['']*len(g.actions_codes)
        for A,idx  in g.actions_codes.items():
            g.rev_action_codes[idx] = A
        g.actions_size   = len(g.actions_codes)  
        g.lexicon_size   = len(g.word_codes)

        g.model = load_model(os.path.join(dirname,'model.prm'))
        return g
        
            
if __name__ == '__main__':
    
    eagerp = ArcEagerGenerativeParser()
        
    treebank = []
    istream = open('UD_English/en-ud-train.conllu')
    dtree = DependencyTree.read_tree(istream)
    idx = 0
    while dtree != None:
        if dtree.is_projective():
            treebank.append(dtree)
            idx += 1
            if idx > 30:
                break
        dtree = DependencyTree.read_tree(istream)
    istream.close()
    df = eagerp.static_nn_train(treebank,max_epochs=100,lr=0005,hidden_dropout=0.3,glove_file='glove/glove.6B.100d.txt')
    #print(df)
    eagerp.save_parser('test')
    eagerp = ArcEagerGenerativeParser.load_parser('test')
    eagerp.test_and_print(treebank)
    eagerg = ArcEagerGenerator.load_generator('test')
    for _ in range(10):
        df = eagerg.generate_sentence(lex_stats=True)
        print(df)
    
