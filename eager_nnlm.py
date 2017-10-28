#! /usr/bin/env python


import os
import os.path
import pickle
import numpy as np
import pandas as pd
import dynet_config
dynet_config.set_gpu()
import dynet as dy

from math import log,exp
from random import shuffle
from numpy.random import choice,rand
from collections import Counter
from lm_utils import NNLMGenerator
from dataset_utils import DependencyTree,UDtreebank_reader
    
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


class ArcEagerGenerativeParser:

    """
    An arc eager language model with local training.

    Designed to run on a GPU.
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
    IOS_TOKEN     = "__IOS__"
    EOS_TOKEN     = "__EOS__"
    UNKNOWN_TOKEN = "__UNK__"

    def __init__(self,embedding_size=300,hidden_size=300,tied_embeddings=True,parser_class='basic'):
        """
        @param embedding_size: size of the embeddings
        @param hidden_size: size of the hidden layer
        @param tied_embeddings: uses weight tying for input and output embedding matrices
        @param parser_class {'basic','extended','star-extended'} controls the number of sensors
        """
        assert(parser_class in ['basic','extended','star-extended'])

        #sets default generic params
        self.model            = None
        self.stack_length     = 3
        self.parser_class     = parser_class
        if self.parser_class == 'basic':
            self.node_size        = 1                                #number of x-values per stack node
        elif self.parser_class == 'extended':
            self.node_size        = 3 
        elif self.parser_class == 'star-extended':
            self.node_size        = 5
            
        self.input_length     = (self.stack_length+1)*self.node_size # stack (+1 = focus node)
        self.tied             = tied_embeddings

        self.actions_size     = 0 
        self.lexicon_size     = 0 
        self.embedding_size   = embedding_size
        self.hidden_size      = hidden_size

        self.word_codes       = None  
        self.actions_codes    = None 
        self.rev_action_codes = None

        
    def __str__(self):
        s = ['Stack size       : %d'%(self.input_length),\
            'Node  size        : %d'%(self.node_size),\
            'Embedding size    : %d'%(self.embedding_size),\
            'Hidden layer size : %d'%(self.hidden_size),\
            'Actions size      : %d'%(self.actions_size),\
            'Lexicon size      : %d'%(self.lexicon_size),\
            'Tied Embeddings   : %r'%(self.tied)]
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
    def code_symbols(self,treebank,lexicon_size=9998):
        """
        Codes lexicon (x-data) and the list of action (y-data)
        on integers.
        @param treebank    : the treebank where to extract the data from
        @param lexicon_size: caps the lexicon to some vocabulary size (default = mikolov size)
        """
        #lexical coding
        lexicon = [ArcEagerGenerativeParser.IOS_TOKEN,\
                   ArcEagerGenerativeParser.EOS_TOKEN,\
                   ArcEagerGenerativeParser.UNKNOWN_TOKEN]
        lex_counts = Counter()
        for dtree in treebank:
            counter.update(dtree.tokens)
        self.lexicon = [w for w,c in lex_counts.most_common(9998-3)]+lexicon
        self.rev_word_codes = list(lexicon)
        self.lexicon_size = len(lexicon)
        self.word_codes = dict([(s,idx) for (idx,s) in enumerate(self.rev_word_codes)])
        
        #structural coding
        actions = [ArcEagerGenerativeParser.LEFTARC,\
                   ArcEagerGenerativeParser.RIGHTARC,\
                   ArcEagerGenerativeParser.PUSH,\
                   ArcEagerGenerativeParser.REDUCE,\
                   ArcEagerGenerativeParser.TERMINATE]
                   #Generate action is implied
                   
        self.rev_action_codes = actions                   
        self.actions_codes = dict([(s,idx) for (idx,s) in enumerate(actions)])
        self.actions_size  = len(actions) 
        

    def read_glove_embeddings(self,glove_filename):
        """
        Reads embeddings from a glove filename and returns an embedding
        matrix for the parser vocabulary.
        @param glove_filename: the file where to read embeddings from
        @return an embedding matrix that can initialize an Embedding layer
        """
        print('Reading embeddings from %s ...'%glove_filename)

        embedding_matrix = (rand(self.lexicon_size,self.embedding_size) - 0.5)/10.0 #uniform init [-0.05,0.05]

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

            
    def make_representation(self,config,action,sentence,structural=True):
        """
        Turns a configuration into a couple of vectors (X,Y) and
        outputs the coded configuration as a tuple of index vectors.
        @param configuration: a parser configuration
        @param action: the ref action code (as a string) or None if the ref action is not known
        @param structural : bool, switch between structural action (True) and lexical action (False)
        @param sentence: a list of tokens (strings)
        @return a couple (X,Y) or just X if no action is given as param
        """        
        S,F,B,A,score = config
        X  = [ArcEagerGenerativeParser.IOS_TOKEN] * self.input_length
        Ns = len(S)
        unk_token = self.word_codes[ArcEagerGenerativeParser.UNKNOWN_TOKEN]

        
        if F is not None: X[0] = self.word_codes.get(sentence[F.root],unk_token)      
        if Ns > 0 :       X[1] = self.word_codes.get(sentence[S[-1].root],unk_token)  
        if Ns > 1 :       X[2] = self.word_codes.get(sentence[S[-2].root],unk_token)  
        if Ns > 2 :       X[3] = self.word_codes.get(sentence[S[-3].root],unk_token)  

        if self.node_size > 1 :
            if F is not None :
                X[4] = self.word_codes.get(sentence[F.ilc],unk_token)      
                X[5] = self.word_codes.get(sentence[F.irc],unk_token)
            if Ns > 0 :   
                X[6] = self.word_codes.get(sentence[S[-1].ilc],unk_token)   
                X[7] = self.word_codes.get(sentence[S[-1].irc],unk_token)
            if Ns > 1 :
                X[8] = self.word_codes.get(sentence[S[-2].ilc],unk_token)  
                X[9] = self.word_codes.get(sentence[S[-2].irc],unk_token)
            if Ns > 2 :  
                X[10] = self.word_codes.get(sentence[S[-3].ilc],unk_token)  
                X[11] = self.word_codes.get(sentence[S[-3].irc],unk_token)
            
        if self.node_size > 3:
            if F is not None :
                X[12] = self.word_codes.get(sentence[F.starlc],unk_token)
                X[13] = self.word_codes.get(sentence[F.starrc()],unk_token)   
            if Ns > 0 :
                X[14] = self.word_codes.get(sentence[S[-1].starlc],unk_token)    
                X[15] = self.word_codes.get(sentence[S[-1].starrc()],unk_token)
            if Ns > 1 :
                X[16] = self.word_codes.get(sentence[S[-2].starlc],unk_token)   
                X[17] = self.word_codes.get(sentence[S[-2].starrc()],unk_token)
            if Ns > 2 :
                X[18] = self.word_codes.get(sentence[S[-3].starlc],unk_token)  
                X[19] = self.word_codes.get(sentence[S[-3].starrc()],unk_token)  

        if action is None:
            return X
        else:
            Y = self.actions_codes[action] if structural else self.word_codes.get(action,unk_token)        
            return (X,Y)

    def make_data_generators(self,treebank,batch_size):
        """
        This returns two data generators suitable for use with dynet.
        One for the lexical submodel and one for the structural submodel
        @param treebank: the treebank (list of sentences) to encode
        @param batch_size: the size of the batches yielded by the generators
        @return (lexical generator, structural generator) as NNLM generator objects
        """
        X_lex    = []
        Y_lex    = []
        X_struct = []
        Y_struct = []
       
        for dtree in treebank:
            Deriv = self.static_oracle_derivation(dtree)
            for (config,action,sentence) in Deriv:
                if type(action) == tuple: #lexical action                    
                    x,y    = self.make_representation(config,action,sentence,structural=False)
                    X_lex.append(x)
                    Y_lex.append(y)
                else:                     #structural action
                    x,y    = self.make_representation(config,action,sentence,structural=True)
                    X_struct.append(x)
                    Y_struct.append(y)
                    
        lex_generator    = NNLMGenerator(X_lex,Y_lex,batch_size)
        struct_generator = NNLMGenerator(X_struct,Y_struct,batch_size)
        return ( lex_generator , struct_generator )


    def predict_logprobs(self,X,Y,structural=True,hidden_out=False):
        """
        Returns the log probabilities of the predictions for this model (batched version).

        @param X: the input indexes from which to predict (each xdatum is expected to be an iterable of integers) 
        @param Y: a list of references indexes for which to extract the prob
        @param structural: switches between structural and lexical logprob evaluation
        @param hidden_out: outputs an additional list of hidden dimension vectors
        @return the list of predicted logprobabilities for each of the provided ref y in Y
        """
        assert(len(X) == len(Y))
        assert(all(len(x) == self.input_length for x in X))

        if structural:
            dy.renew_cg()
            W = dy.parameter(self.hidden_weights)
            E = dy.parameter(self.input_embeddings)
            A = dy.parameter(self.action_weights)
            
            batched_X  = zip(*X) #transposes the X matrix
            embeddings = [dy.pick(E, xcolumn) for xcolumn in batched_X]
            xdense     = dy.concatenate(embeddings)
            preds      = dy.pickneglogsoftmax(A * dy.tanh( W * xdense ),Y)
            dy.forward(preds)
            return [-ypred.value()  for ypred in preds]

        else:#lexical
            if self.tied:
                dy.renew_cg()
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.embedding_matrix)
                batched_X  = zip(*X) #transposes the X matrix
                embeddings = [dy.pick(E, xcolumn) for xcolumn in X]
                xdense     = dy.concatenate(embeddings)
                preds      = dy.pickneglogsoftmax(E * dy.tanh( W * xdense ),Y)
                dy.forward(preds)
                return [-ypred.value()  for ypred in preds]
            else:
                dy.renew_cg()
                O = dy.parameter(self.output_weights)
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.embedding_matrix)
                batched_X  = zip(*X) #transposes the X matrix
                embeddings = [dy.pick(E, xcolumn) for xcolumn in X]
                xdense     = dy.concatenate(embeddings)
                preds      = dy.pickneglogsoftmax(O * dy.tanh( W * xdense ),Y)
                dy.forward(preds)
                return [-ypred.value()  for ypred in preds]

    
    def static_train(self,\
                    train_treebank,\
                    validation_treebank,\
                    lr=0.001,\
                    hidden_dropout=0.1,\
                    batch_size=64,\
                    max_epochs=100,\
                    max_lexicon_size=9998,\
                    glove_file=None):
        """
        Locally trains a model with a static oracle and a multi-task standard feedforward NN.  
        @param train_treebank      : a list of dependency trees
        @param validation_treebank : a list of dependency trees
        @param lr                  : learning rate
        @param hidden_dropout      : dropout on hidden layer
        @param batch_size          : size of mini batches
        @param max_epochs          : max number of epochs
        @param max_lexicon_size    : max number of entries in the lexicon
        @param glove_file          : file where to find pre-trained word embeddings   
        """
        print("Encoding dataset from %d trees."%len(train_treebank))

        #(1) build dictionaries
        self.code_symbols(train_treebank,lexicon_size = max_lexicon_size)

        #(2) encode data sets
        lex_train_gen , struct_train_gen  = self.make_data_generators(train_treebank,batch_size)
        lex_dev_gen   , struct_dev_gen    = self.make_data_generators(train_treebank,batch_size)
        
        print(self)
        print("training examples [N] = %d\nBatch size = %d\nDropout = %f\nlearning rate = %f"%(len(Y),batch_size,hidden_dropout,lr))

        #(3) make network
        self.model = dy.ParameterCollection()
        self.hidden_weights   = self.model.add_parameters((self.hidden_size,self.embedding_size*self.input_length))
        self.action_weights   = self.model.add_parameters((self.actions_size,self.hidden_size))
        if glove_file is None:
            self.input_embeddings  = self.model.add_parameters((self.lexicon_size,self.embedding_size))
        else:
            self.input_embeddings  = self.model.parameters_from_numpy(self.read_glove_embeddings(glove_file))
        if not self.tied:
            self.output_embeddings = self.model.add_parameters((self.lexicon_size,self.hidden_size))

        #(4) fitting
        lex_gen       = lex_train_gen.next_batch()
        struct_gen    = struct_train_gen.next_batch()
        max_batches = max( lex_gen.get_num_batches(), struct_gen.get_num_batches() )

        lex_valid_gen       = lex_dev_gen.next_batch()
        struct_valid_gen    = struct_dev_gen.next_batch()
        
        min_nll = float('inf')
        trainer = dy.AdamTrainer(self.model,alpha=lr)
        history_log = []
        for e in range(max_epochs):
            struct_loss,lex_loss = 0,0
            struct_N,lex_N       = 0,0
            start_t = time.time()
            for b in range(max_batches):
                #struct
                X_struct,Y_struct = next(struct_gen)
                #question of proportions : should struct and gen be evenly sampled or not (??)
                dy.renew_cg()
                W = dy.parameter(self.hidden_weights)
                E = dy.parameter(self.input_embeddings)
                A = dy.parameter(self.action_weights)
                batched_X        = zip(*X_struct)  #transposes the X matrix
                lookups          = [dy.pick_batch(E,xcolumn) for xcolumn in batched_X]
                xdense           = dy.concatenate(lookups)
                ybatch_preds     = dy.pickneglogsoftmax_batch(A * dy.dropout(dy.tanh( W * xdense ),hidden_dropout),Y)
                loss             = dy.sum_batches(ybatch_preds)
                struct_N         +=len(Y_struct)
                struct_loss      += loss.value()
                loss.backward()
                trainer.update()
                #lex
                X_lex,Y_lex = next(lex_gen)
                if self.tied:
                    dy.renew_cg()
                    W = dy.parameter(self.hidden_weights)
                    E = dy.parameter(self.input_embeddings)
                    batched_X        = zip(*X_lex) #transposes the X matrix
                    lookups          = [dy.pick_batch(E,xcolumn) for xcolumn in batched_X]
                    xdense           = dy.concatenate(lookups)
                    ybatch_preds     = dy.pickneglogsoftmax_batch(E * dy.dropout(dy.tanh( W * xdense ),hidden_dropout),Y)
                    loss             = dy.sum_batches(ybatch_preds)
                else:
                    dy.renew_cg()
                    W = dy.parameter(self.hidden_weights)
                    E = dy.parameter(self.input_embeddings)
                    O = dy.parameter(self.output_embeddings)
                    batched_X        = zip(*X_lex) #transposes the X matrix
                    lookups          = [dy.pick_batch(E,xcolumn) for xcolumn in batched_X]
                    xdense           = dy.concatenate(lookups)
                    ybatch_preds     = dy.pickneglogsoftmax_batch(O * dy.dropout(dy.tanh( W * xdense ),hidden_dropout),Y)
                    loss             = dy.sum_batches(ybatch_preds)
                lex_N            +=len(Y_lex)
                lex_loss         += loss.value()
                loss.backward()
                trainer.update()
            end_t = time.time()
            # (5) validation
            X_lex_valid,Y_lex_valid       = lex_valid_gen.batch_all()
            lex_valid_nll = -sum(self.predict_logprobs(X_lex_valid,Y_lex_valid,structural=False))
            
            X_struct_valid,Y_struct_valid = struct_valid_gen.batch_all()
            struct_valid_nll = -sum(self.predict_logprobs(X_struct_valid,Y_struct_valid,structural=True))
            
            history_log.append((e,end_t-start_t,lex_loss,struct_loss,lex_valid_nll,struct_valid_nll,lex_valid_nll+struct_valid_nll))
            print('Epoch %d (%.2f sec.) NLL_lex (train) = %f, NLL_struct (train) = %f, NLL_lex (valid) = %f, NLL_struct (valid) = %f, NLL_all (valid) = %f'%tuple(history_log[-1]),flush=True)
            if  lex_valid_nll+struct_valid_nll < min_nll:
                pass #auto-save model
            
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
        
    
    # def generate_sentence(self,max_len=2000,lex_stats=False):
    #     """
    #     @param lex_stats: generate a table with word,log(prefix_prob),log(local_prob),num_actions
    #     """
    #     C = self.init_configuration()
    #     derivation = [C]
    #     action,score =  self.stochastic_oracle(C,is_first_action=True)
    #     stats = []
    #     while action != ArcEagerGenerator.TERMINATE and len(derivation) < max_len:
            
    #         if action == ArcEagerGenerator.LEFTARC:
    #             C = self.leftarc(C,score)
    #         elif action == ArcEagerGenerator.RIGHTARC:
    #             C = self.rightarc(C,score)
    #         elif action == ArcEagerGenerator.REDUCE:
    #             C = self.reduce_config(C,score)
    #         elif action == ArcEagerGenerator.PUSH:
    #             C = self.push(C,score)
    #         else:
    #             action, w = action
    #             assert(action == ArcEagerGenerator.GENERATE)
    #             C = self.generate(C,w,score)
    #             stats.append((w,C[4],log(score),len(derivation)))
    #         derivation.append(C)

    #         action,score = self.stochastic_oracle(C)
            
    #     if lex_stats:
    #         df = pd.DataFrame(stats,columns=['word','log(P(deriv_prefix))','log(P(local))','nActions'])
    #         return df
    #     else:
    #         return derivation

    
    # #CODING & SCORING
    # def stochastic_oracle(self,configuration,is_first_action=False):
    #     S,F,terminals,A,score = configuration
    #     X = np.array([self.make_representation(configuration)])
    #     Y = self.model.predict(X,batch_size=1)[0]
        
    #     def has_governor(node_idx,arc_list):
    #         """
    #         Checks if a node has a governor
    #         @param node_idx: the index of the node
    #         @arc_list: an iterable over arc tuples
    #         @return a boolean
    #         """
    #         return any(node_idx  == didx for (gidx,didx) in arc_list)
        
    #     if is_first_action:#otherwise predicts terminate with p=1.0 because init config is also the config right before calling terminate
    #         while True:
    #             action_code = choice(self.actions_size)#uniform draw
    #             action_score = 1.0/self.actions_size
    #             action = self.rev_action_codes[action_code]
    #             if type(action) == tuple:#this is a generate action
    #                 return (action,action_score)            
    #     if not S or F is None or S[-1].root == 0 or not has_governor(S[-1].root,A):
    #         Y[self.actions_codes[ArcEagerGenerator.LEFTARC]] = 0.0
    #     if not S or F is None:
    #         Y[self.actions_codes[ArcEagerGenerator.RIGHTARC]] = 0.0
    #     if F is None or not S or not has_governor(S[-1].root,A):
    #         Y[self.actions_codes[ArcEagerGenerator.REDUCE]] = 0.0
    #     if F is None:
    #         Y[self.actions_codes[ArcEagerGenerator.PUSH]] = 0.0
    #     if F is not None:
    #         la = Y[self.actions_codes[ArcEagerGenerator.LEFTARC]]
    #         ra = Y[self.actions_codes[ArcEagerGenerator.RIGHTARC]]
    #         r  = Y[self.actions_codes[ArcEagerGenerator.REDUCE]]
    #         p  = Y[self.actions_codes[ArcEagerGenerator.PUSH]]
    #         t  = Y[self.actions_codes[ArcEagerGenerator.TERMINATE]]
            
    #         Y = np.zeros(self.actions_size)
            
    #         Y[self.actions_codes[ArcEagerGenerator.LEFTARC]]   = la
    #         Y[self.actions_codes[ArcEagerGenerator.RIGHTARC]]  = ra
    #         Y[self.actions_codes[ArcEagerGenerator.REDUCE]]    = r
    #         Y[self.actions_codes[ArcEagerGenerator.PUSH]]      = p
    #         Y[self.actions_codes[ArcEagerGenerator.TERMINATE]] = t
            
    #     Z = Y.sum()
    #     if Z == 0.0:#no action possible, trapped in dead-end, abort.
    #         return (ArcEagerGenerator.TERMINATE,np.finfo(float).eps)
            
    #     Y /= Z
    #     action_code = choice(self.actions_size,p=Y)
    #     action_score = Y[action_code]
    #     action = self.rev_action_codes[action_code]

    #     #print distribution:
    #     #print ('kbest')
    #     #kbest = sorted([(p,idx) for (idx,p) in enumerate(Y)],reverse=True)[:20]
    #     #for p,idx in kbest:
    #     #    print(idx,self.rev_action_codes[idx],p)
                
    #     return (action,action_score)


if __name__ == '__main__':
    
        
    train_treebank = UDtreebank_reader('ptb/ptb_deps.train',tokens_only=False)
    dev_treebank = UDtreebank_reader('ptb/ptb_deps.dev',tokens_only=False)
    
    eagerp = ArcEagerGenerativeParser()
    eagerp.static_train(train_treebank[:20],dev_treebank[:20],glove_file='glove/glove.6B.300d.txt')
