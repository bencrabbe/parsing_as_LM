import dynet as dy
import numpy as np
import numpy.random as npr
import getopt
import warnings
import json

from collections import Counter
from random import shuffle
from constree import *
from lexicons import *
from proc_monitors import *
from rnng_params import *

class StackSymbol:
    """
    A convenience class for stack symbols
    """
    PREDICTED = 1
    COMPLETED = 0
    
    def __init__(self,symbol,status,embedding):
        """
        @param symbol: a non terminal or a word
        @param status : predicted or completed
        @param embedding : a dynet expression being the embedding of the subtree dominated by this symbol (or word)
        """
        self.symbol,self.status,self.embedding = symbol,status,embedding

    def copy(self):
        return StackSymbol(self.symbol,self.status,self.embedding)

    def complete(self):
        self.status = StackSymbol.COMPLETED
         
    def __str__(self):
        s =  '*%s'%(self.symbol,) if self.status == StackSymbol.PREDICTED else '%s*'%(self.symbol,)
        return s


class BeamElement:
    """
    Class used for storing elements in the parser's beam
    """         
    #representation used for lazy delayed exec of actions in the beam
    def __init__(self,prev_item,current_action,local_score):
        self.prev_element           = prev_item               #prev beam item (history)
        self.structural_history     = None                    #scheduling info
        self.incoming_action        = current_action
        self.config                 = None           
        self.local_score            = local_score
        self.deriv_length           = self.prev_element.deriv_length+1 if self.prev_element else 1
        
    def update_history(self,update_val = None):
        if update_val is None:
            #no copy in case the action is not structural
            self.structural_history = self.prev_element.structural_history
        else:
            #copy in case the action **is** structural
            self.structural_history = self.prev_element.structural_history + [update_val]
                
    @staticmethod
    def figure_of_merit(elt):
        #provides a score for ranking the elements in the beam
        #could add derivation length for further normalization (?)
        _,_,_,_,lab_state,prefix_score = elt.prev_element.config
        return (elt.local_score + prefix_score)#/elt.deriv_length
    
#Monitoring loss & accurracy
class OptimMonitor:
    def __init__(self):
        self.reset_all()
        self.ppl_dataset = []
        
    def display_NLL_log(self,tree_idx=None,reset=False):
        global_nll = self.lex_loss+self.struct_loss+self.nt_loss
        N =  self.lexN + self.structN + self.ntN
        if N == 0:
            return
        if tree_idx:
            sys.stdout.write("\rTree #%d Mean NLL : %.5f, PPL : %.5f, Lex-PPL : %.5f, NT-PPL : %.5f, Struct-PPL: %.5f\n"%(tree_idx,\
                                                                                                                        global_nll/N,\
                                                                                                                        np.exp(global_nll/N),\
                                                                                                                        np.exp(self.lex_loss/self.lexN),
                                                                                                                        np.exp(self.nt_loss/self.ntN),
                                                                                                                        np.exp(self.struct_loss/self.structN)))
        else:
            sys.stdout.write("\rMean NLL : %.5f, PPL : %.5f, Lex-PPL : %.5f, NT-PPL : %.5f, Struct-PPL: %.5f\n"%(global_nll/N,\
                                                                                                              np.exp(global_nll/N),\
                                                                                                              np.exp(self.lex_loss/self.lexN),
                                                                                                              np.exp(self.nt_loss/self.ntN),
                                                                                                              np.exp(self.struct_loss/self.structN)))
        sys.stdout.flush()
        self.ppl_dataset.append((np.exp(global_nll/N),\
                                np.exp(self.lex_loss/self.lexN),\
                                np.exp(self.struct_loss/self.structN),\
                                np.exp(self.nt_loss/self.ntN),\
                                self.lexN,\
                                self.structN,\
                                self.ntN))

        if reset:
            self.reset_loss_counts()
        
    def display_ACC_log(self,reset=False):
        global_acc = self.struct_acc+self.lex_acc+self.nt_acc
        N = self.acc_lexN+self.acc_structN+self.acc_ntN 
        if N == 0:
            return
        sys.stdout.write("\nMean ACC : %.5f, Lex acc : %.5f, Struct acc : %.5f, NT acc : %.5f\n"%(global_acc/N,\
                                                                                                    self.lex_acc/self.acc_lexN,\
                                                                                                    self.struct_acc/self.acc_structN,\
                                                                                                    self.nt_acc/self.acc_ntN))
        sys.stdout.flush()
        if reset:
            self.reset_acc_counts()
        
    def reset_all(self):
        self.reset_loss_counts()
        self.reset_acc_counts()
        
    def reset_loss_counts(self):
        self.lex_loss    = 0
        self.struct_loss = 0
        self.nt_loss     = 0
        self.lexN,self.structN,self.ntN = 0,0,0

    def reset_acc_counts(self):
        self.lex_acc    = 0
        self.struct_acc = 0
        self.nt_acc     = 0
        self.acc_lexN,self.acc_structN,self.acc_ntN = 0,0,0
        
        
    def save_loss_curves(self,filename):
        ostream = open(filename,'w')
        print(','.join(['ppl','lex-ppl','struct-ppl','nt-ppl','N-lex','N-struct','N-nt']),file=ostream)
        for line in self.ppl_dataset:
            print(",".join([str(elt) for elt in line]),file=ostream)
        ostream.close()
        
    def add_ACC_datum(self,datum_correct,configuration):
        """
        Accurracy logging
        @param datum_correct: last prediction was correct ?
        @param configuration : the configuration used to make the prediction
        """
        S,B,n,stack_state,datum_type,score = configuration
        if datum_type  == RNNGparser.WORD_LABEL:
            self.lex_acc  += datum_correct
            self.acc_lexN += 1
        elif datum_type == RNNGparser.NT_LABEL:
            self.nt_acc  += datum_correct
            self.acc_ntN += 1
        elif datum_type == RNNGparser.NO_LABEL:
            self.struct_acc  += datum_correct
            self.acc_structN += 1

            
    def add_NLL_datum(self,datum_loss,configuration):
        """
        NLL logging
        @param datum_loss: the -logprob of the correct action
        @param configuration : the configuration used to make the prediction
        """
        S,B,n,stack_state,datum_type,score = configuration

        datum_loss = - max(-datum_loss,np.log(np.finfo(float).eps)) #smoothes potential zeroes
        
        if datum_type  == RNNGparser.WORD_LABEL:
            self.lex_loss += datum_loss
            self.lexN     += 1
            
        elif datum_type == RNNGparser.NT_LABEL:
            self.nt_loss += datum_loss
            self.ntN     +=1
            
        elif datum_type == RNNGparser.NO_LABEL:
            self.struct_loss += datum_loss
            self.structN     +=1
            
class RNNGparser:
    """
    This is RNNG with in-order tree traversal.
    """        
    #action codes
    SHIFT           = '<S>'
    OPEN            = '<O>'
    CLOSE           = '<C>'
    TERMINATE       = '<T>'
    
    #labelling states
    WORD_LABEL      = '@w'
    NT_LABEL        = '@n'
    NO_LABEL        = '@'
    
    #special tokens
    UNKNOWN_TOKEN = '<UNK>'
    START_TOKEN   = '<START>'

    def __init__(self,max_vocabulary_size=10000,
                 stack_embedding_size=50,
                 stack_memory_size=50):
        """
        @param max_vocabulary_size     : max number of words in the vocab
        @param stack_embedding_size    : size of stack lstm input 
        @param stack_memory_size       : size of the stack and tree lstm hidden layers
        """
        self.max_vocab_size       = max_vocabulary_size
        self.stack_embedding_size = stack_embedding_size
        self.stack_hidden_size    = stack_memory_size
        self.dropout              = 0.0
        #Extras (brown lexicon and external embeddings)
        self.blex = None
        self.ext_embeddings = False
        #self.tied = False

    #oracle, derivation and trees
    def oracle_derivation(self,config,ref_tree,sentence,struct_history,root=True):
        """
        Returns an oracle derivation given a reference tree
        @param config         : the running configuration
        @param ref_tree       : a ConsTree
        @param sentence       : a list of strings, the tokens
        @param struct_history : the history of structural actions as list
        @return a couple made of a derivation and the current configuration
        """
        if ref_tree.is_leaf():
            if not RNNGparser.SHIFT in [ self.actions[i] for i in self.restrict_structural_actions(config,struct_history) ]:
                print('oracle unsound <shift>')
            config,struct_history = self.move_state(sentence,config,struct_history,RNNGparser.SHIFT,0)
            config,struct_history = self.move_state(sentence,config,struct_history,ref_tree.label,0)
            return ( [RNNGparser.SHIFT, ref_tree.label] , config, struct_history)
        else:
            first_child = ref_tree.children[0]
            derivation, config ,struct_history      = self.oracle_derivation(config,first_child,sentence,struct_history,root=False)
    
            if not RNNGparser.OPEN in [ self.actions[i] for i in self.restrict_structural_actions(config,struct_history) ]:
                print('oracle unsound <open>')
            config,struct_history    = self.move_state(sentence,config,struct_history,RNNGparser.OPEN,0)   
            config,struct_history    = self.move_state(sentence,config,struct_history,ref_tree.label,0)
            derivation.extend([RNNGparser.OPEN,ref_tree.label])
            for child in ref_tree.children[1:]:
                subderivation,config,struct_history = self.oracle_derivation(config,child,sentence,struct_history,root=False) 
                derivation.extend(subderivation)
            if not RNNGparser.CLOSE in [ self.actions[i] for i in self.restrict_structural_actions(config,struct_history) ]:
                print('oracle unsound <close>')
            config,struct_history    =  self.move_state(sentence,config,struct_history,RNNGparser.CLOSE,0)
            derivation.append(RNNGparser.CLOSE)
        if root:
            derivation.append(RNNGparser.TERMINATE)
        return ( derivation, config, struct_history )

    @staticmethod
    def derivation2tree(derivation,tokens):
        """
        Transforms a derivation back to a tree
        @param derivation: a derivation to use for generating a tree
        @tokens : the source tokens of the sentence
        """
        stack = []
        tok_idx   = 0
        prev_action = None
        for action in derivation:
            if prev_action == RNNGparser.SHIFT:
                lex = ConsTree(tokens[tok_idx])
                stack.append((lex,False))
                tok_idx +=1
            elif prev_action == RNNGparser.OPEN:
                lc_child = stack.pop()
                lc_node,status = lc_child
                assert(status==False)
                root = ConsTree(action,children=[lc_node])
                stack.append((root,True))
            elif action == RNNGparser.CLOSE:
                children = []
                while stack:
                    node,status = stack.pop()
                    if status:
                        for c in reversed(children):
                            node.add_child(c)
                        stack.append((node,False))
                        break
                    else:
                        children.append(node)
            prev_action = action
        return stack[0][0]
                
    def pretty_print_configuration(self,configuration):
        S,B,n,stack_state,hist_state,score = configuration

        stack  = ','.join([str(elt) for elt in S])
        bfr    = ','.join([str(elt) for elt in B])
        return '< (%s) , (%s) , %d>'%(stack,bfr,n)


    #coding    
    def code_lexicon(self,treebank,max_vocab_size):
        """
        Codes a lexicon (x-data) on integers indexes.
        @param treebank: the treebank where to extract the data from
        @param max_vocab_size: the upper bound on the size of the vocabulary
        """
        lexicon = Counter()
        for tree in treebank:
            sentence = tree.tokens(labels=True) 
            lexicon.update(sentence)
        self.lexicon = SymbolLexicon(lexicon,unk_word=RNNGparser.UNKNOWN_TOKEN,special_tokens=[RNNGparser.START_TOKEN],max_lex_size=max_vocab_size)
        
    def code_nonterminals(self,treebank):
        """
        Extracts the nonterminals from a treebank.
        @param treebank: the treebank where to extract the data from
        """
        nonterminals = set([])
        for tree in treebank:
            nonterminals.update(tree.collect_nonterminals())
        #print(nonterminals)
        self.nonterminals = SymbolLexicon(list(nonterminals),unk_word=None,special_tokens=[])
        #print(self.nonterminals)
        return self.nonterminals

    def code_struct_actions(self):
        """
        Codes the structural actions on integers
        """
        self.actions         = [RNNGparser.SHIFT,RNNGparser.OPEN,RNNGparser.CLOSE,RNNGparser.TERMINATE]
        self.action_codes    = dict([(sym,idx) for (idx,sym) in enumerate(self.actions)])
        self.open_mask       = np.array([True,False,True,True]) #we assume scores are log probs
        self.shift_mask      = np.array([False,True,True,True])
        self.close_mask      = np.array([True,True,False,True])
        self.terminate_mask  = np.array([True,True,True,False])

        
    #transition system
    def init_configuration(self,N):
        """
        A: configuration is a 6tuple (S,B,n,stack_mem,labelling?,sigma)
        
        S: is the stack
        B: the buffer
        n: the number of predicted constituents in the stack
        stack_mem: the current state of the stack lstm
        lab_state: the labelling state of the configuration
        sigma:     the *prefix* score of the configuration. I assume scores are log probs
        
        Creates an initial configuration, with empty stack, full buffer (as a list of integers and null score)
        Stacks are filled with non terminals of type strings and/or terminals of type integer.
        Buffers are only filled with terminals of type integers.
        
        @param N: the length of the input sentence
        @return a configuration
        """
        stackS = self.stack_rnn.initial_state()

        word_idx = self.lexicon.index(RNNGparser.START_TOKEN)
        word_embedding = self.lex_embedding_matrix[word_idx]
        stackS = stackS.add_input(word_embedding)

        return ([],tuple(range(N)),0,stackS,RNNGparser.NO_LABEL,0.0)

    def shift_action(self,configuration,local_score):
        """
        That's the structural RNNG SHIFT action.
        @param configuration : a configuration tuple
        @param local_score: the local score of the action (logprob)
        @return a configuration resulting from shifting the next word into the stack 
        """    
        S,B,n,stack_state,lab_state,score = configuration
        return (S,B,n,stack_state,RNNGparser.WORD_LABEL,score+local_score)
    
    def word_action(self,configuration,sentence,local_score):
        """
        That's the word labelling action (implements a traditional shift).
        @param configuration : a configuration tuple
        @param sentence: the list of words of the sentence as a list of string tokens
        @param local_score: the local score of the action (logprob)
        @return a configuration resulting from shifting the next word into the stack 
        """
        S,B,n,stack_state,lab_state,score = configuration
        #print(sentence[B[0]],self.lexicon.normal_wordform(sentence[B[0]]))
        word_idx = self.lexicon.index(sentence[B[0]])  
        E = dy.parameter(self.lex_embedding_matrix)
        word_embedding = self.rnng_dropout(E[word_idx]) 
        word_embedding = self.rnng_nobackprop(word_embedding,sentence[B[0]]) #we do not want to backprop when using external embeddings
        return (S + [StackSymbol(B[0],StackSymbol.COMPLETED,word_embedding)],B[1:],n,stack_state.add_input(word_embedding),RNNGparser.NO_LABEL,score+local_score)

    def open_action(self,configuration,local_score):
        """
        That's the structural RNNG OPEN action.
        @param configuration : a configuration tuple
        @param local_score: the local score of the action (logprob)
        @return a configuration resulting from opening the constituent
        """
        S,B,n,stack_state,lab_state,score = configuration
        return (S,B,n,stack_state,RNNGparser.NT_LABEL,score+local_score)
    
    def nonterminal_action(self,configuration,X,local_score):
        """
        That's the non terminal labelling action.
        @param configuration : a configuration tuple
        @param X: the label of the nonterminal 
        @param local_score: the local score of the action (logprob)
        @return a configuration resulting from labelling the nonterminal
        """
        S,B,n,stack_state,lab_state,score = configuration

        nt_idx = self.nonterminals.index(X)
        nonterminal_embedding = self.rnng_dropout(self.nt_embedding_matrix[nt_idx])

        #We backup the current stack top
        first_child = S[-1]
        #We pop the first child, push the non terminal and re-push the first child 
        stack_state = stack_state.prev()
        stack_state = stack_state.add_input(nonterminal_embedding)
        stack_state = stack_state.add_input(first_child.embedding)
        return (S[:-1] + [StackSymbol(X,StackSymbol.PREDICTED,nonterminal_embedding),first_child],B,n+1,stack_state,RNNGparser.NO_LABEL,score+local_score)

    
    def close_action(self,configuration,local_score):
        """
        That's the RNNG CLOSE action.
        @param configuration : a configuration tuple
        @param local_score: the local score of the action (logprob)
        @return a configuration resulting from closing the current constituent
        """
        S,B,n,stack_state,lab_state,score = configuration
        assert( n > 0 )
        
        #Finds the closest predicted constituent in the stack and backtracks the stack lstm.
        midx = -1
        for idx,symbol in enumerate(reversed(S)):
            if symbol.status == StackSymbol.PREDICTED:
                root_idx = len(S)-idx-1
                break
            else:
                stack_state = stack_state.prev()
        stack_state = stack_state.prev()
        root_symbol = S[root_idx].copy()
        root_symbol.complete()
        children    = S[root_idx+1:]
            
        #compute the tree embedding with the tree_rnn
        nt_idx = self.nonterminals.index(root_symbol.symbol)
        NT_embedding = self.rnng_dropout(self.nt_embedding_matrix[nt_idx])
        s1 = self.fwd_tree_rnn.initial_state()
        s1 = s1.add_input(NT_embedding)
        for c in children:
            s1 = s1.add_input(c.embedding)
        fwd_tree_embedding = s1.output()
        s2 = self.bwd_tree_rnn.initial_state()
        s2 = s2.add_input(NT_embedding)
        for c in reversed(children):
            s2 = s2.add_input(c.embedding)
        bwd_tree_embedding = s2.output()
        
        x = dy.concatenate([fwd_tree_embedding,bwd_tree_embedding])
        W = dy.parameter(self.tree_rnn_out)
        tree_embedding = self.rnng_dropout(dy.tanh(W * x))
        return (S[:root_idx]+[root_symbol],B,n-1,stack_state.add_input(tree_embedding),RNNGparser.NO_LABEL,score+local_score)

    def restrict_structural_actions(self,configuration,structural_history):
        """ 
        This returns a list of integers stating which structural actions are legal for a given configuration
        @param configuration: the current configuration
        @param structural_history: the structural actions history.
        @return a list of integers, indexes of legal actions
        """
        #Assumes masking log probs
        MASK = np.array([True] * len(self.actions))
        S,B,n,stack_state,lab_state,local_score = configuration
        
        hist_1  = structural_history[-1]
        hist_2  = structural_history[-2] if len(structural_history) >= 2 else None
        
        if not B or not S or hist_1 == RNNGparser.OPEN:
            MASK *= self.open_mask
        if B or n > 0 or len(S) > 1:
            MASK *= self.terminate_mask
        if not B:
            MASK *= self.shift_mask
        if not S or n == 0 or (hist_1 == RNNGparser.OPEN and hist_2 != RNNGparser.SHIFT):
            MASK *= self.close_mask

        restr_list = [idx for idx,mval in enumerate(MASK) if mval]
        return restr_list
    
        
    #scoring & representation system
    def make_structure(self,w2vfilename=None):
        """
        Allocates the network structure
        @param w2filename: an external word embedding dictionary
        """
        actions_size = len(self.actions)
       
        #Model structure
        self.model                 = dy.ParameterCollection()

        #symbols & word embeddings (some setups may affect the structure of the network)
        self.nt_embedding_matrix   = self.model.add_lookup_parameters((self.nonterminals.size(),self.stack_embedding_size),init='glorot') #symbols embeddings

        if not w2vfilename:
            self.lex_embedding_matrix  = self.model.add_lookup_parameters((self.lexicon.size(),self.stack_embedding_size),init='glorot')
        else :
            print('Using external embeddings.',flush=True)                                                          #word embeddings
            self.ext_embeddings       =  True

            W,M = RNNGparser.load_embedding_file(w2vfilename)
            embed_dim = M.shape[1]
            self.stack_embedding_size = embed_dim
            E = self.init_ext_embedding_matrix(W,M)
            self.lex_embedding_matrix = self.model.lookup_parameters_from_numpy(E)

        #top level task predictions
        self.struct_out             = self.model.add_parameters((actions_size,self.stack_hidden_size),init='glorot')          #struct action output layer
        self.struct_bias            = self.model.add_parameters((actions_size),init='glorot')

        if self.blex:
            self.lex_out            = self.model.add_parameters((self.blex.size(),self.stack_hidden_size),init='glorot')              #lex action output layer
            self.lex_bias           = self.model.add_parameters((self.blex.size()),init='glorot')
        else:
            self.lex_out            = self.model.add_parameters((self.lexicon.size(),self.stack_hidden_size),init='glorot')          #lex action output layer
            self.lex_bias           = self.model.add_parameters((self.lexicon.size()),init='glorot')

        self.nt_out                 = self.model.add_parameters((self.nonterminals.size(),self.stack_hidden_size),init='glorot')               #nonterminal action output layer
        self.nt_bias                = self.model.add_parameters((self.nonterminals.size()),init='glorot')

        #stack rnn 
        self.stack_rnn             = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)        # main stack rnn
        #tree rnn
        self.fwd_tree_rnn          = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)        # bi-rnn for tree embeddings
        self.bwd_tree_rnn          = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)
        self.tree_rnn_out          = self.model.add_parameters((self.stack_embedding_size,self.stack_hidden_size*2),init='glorot')       # out layer merging the tree bi-rnn output
        
    def rnng_dropout(self,expr):
        """
        That is a conditional dropout that applies dropout to a dynet expression only at training time
        @param expr: a dynet expression
        @return a dynet expression
        """
        if self.dropout == 0:
            return expr
        else:
            return dy.dropout(expr,self.dropout)

    def rnng_nobackprop(self,expr,word_token):
       """
       Function controlling whether to block backprop or not on lexical embeddings
       @param expr: a dynet expression
       @param word_token: the lexical token 
       @return a dynet expression
       """
       if self.ext_embeddings and word_token in self.lexicon:  #do not backprop with external embeddings when word is known to the lexicon
           return dy.nobackprop(expr)
       else:
           return expr
            
        
    def raw_action_distrib(self,configuration,structural_history): #max_prediction=False,ref_action=None,backprop=True):
        """
        This predicts the next action distribution and constrains it given the configuration context.
        @param configuration: the current configuration
        @param structural_history  : the sequence of structural actions performed so far by the parser
        @return a dynet expression
        """        
        S,B,n,stack_state,lab_state,local_score = configuration
        
        if lab_state == RNNGparser.WORD_LABEL:                             #generate wordform action
            
            W = dy.parameter(self.lex_out)
            b = dy.parameter(self.lex_bias)
            return dy.log_softmax(W * self.rnng_dropout(dy.tanh(stack_state.output())) + b)
        
        elif lab_state == RNNGparser.NT_LABEL:                             #generates a non terminal labelling
            W = dy.parameter(self.nt_out)
            b = dy.parameter(self.nt_bias)
            return dy.log_softmax(W * self.rnng_dropout(dy.tanh(stack_state.output())) + b)
            
        else:                                                               #lab_state == RNNGparser.NO_LABEL perform a structural action
            W = dy.parameter(self.struct_out)
            b = dy.parameter(self.struct_bias)
            restr = self.restrict_structural_actions(configuration,structural_history)
            if restr:
                return dy.log_softmax(W * self.rnng_dropout(dy.tanh(stack_state.output())) + b,restr)
            #parse failure (parser trapped)
            warnings.warn('oops. parser trapped (to be fixed)...',RuntimeWarning)
            return None
        
    def predict_action_distrib(self,configuration,structural_history,sentence,max_only=False):
        """
        Predicts the action distribution for testing purposes.
        @param configuration: the current configuration
        @param structural_history  : the sequence of structural actions performed so far by the parser
        @param sentence : a list of tokens as strings
        @param max_only : returns only the couple (action,logprob) with highest score
        
        @return a list of (action,scores) or a single tuple if max_only is True
        """
        S,B,n,stack_state,lab_state,local_score = configuration
        logprobs = self.raw_action_distrib(configuration,structural_history)
        if logprobs == None: #dead end
            if max_only:
                print('sorry, parser trapped. aborting.')
                exit(1)
            return [] #in a beam context parsing can continue...
        logprobs = logprobs.npvalue()
        if lab_state == RNNGparser.WORD_LABEL:        
            next_word = self.lexicon.index(sentence[B[0]]) if not self.blex else self.blex.index(sentence[B[0]])
            score = logprobs[next_word] if not self.blex else logprobs[next_word]+self.blex.word_emission_prob(sentence[B[0]]) 
            if max_only :
                return (next_word,score)
            return [(next_word,score)]         #TODO: CHECK why this returns an integer whereas other switches in the function return a string
        elif lab_state == RNNGparser.NT_LABEL: #label NT action
            if max_only:
                idx = np.argmax(logprobs)
                return (self.nonterminals.wordform(idx),logprobs[idx])
            return list(zip(self.nonterminals.i2words,logprobs))            
        else:                                   #lab_state == RNNGparser.NO_LABEL perform a structural action
            if max_only:
                idx = np.argmax(logprobs)
                return (self.actions[idx],logprobs[idx])
            return [(act,logp) for act,logp in zip(self.actions,logprobs) if logp > -np.inf]

    def backprop_action_distrib(self,configuration,structural_history,ref_action):
        """
        This performs a forward, backward and update pass on the network for this datum.
        @param configuration: the current configuration
        @param structural history: the list of structural actions performed so far
        @param ref_action  : the reference action as a string
        @return : the loss (NLL) for this action
        """
        S,B,n,stack_state,lab_state,local_score = configuration
        logprobs = self.raw_action_distrib(configuration,structural_history)

        if lab_state == RNNGparser.WORD_LABEL:
            ref_prediction = self.lexicon.index(ref_action) if not self.blex else self.blex.index(ref_action)
        elif lab_state == RNNGparser.NT_LABEL:
            ref_prediction = self.nonterminals.index(ref_action)
        else:
            ref_prediction = self.action_codes[ref_action]
            
        loss       = -dy.pick(logprobs,ref_prediction)
        loss_val   = loss.value()
        loss.backward()
        self.trainer.update()
        return loss_val
    
    def eval_action_distrib(self,configuration,structural_history,ref_action):
        """
        This performs a forward pass on the network for this datum and returns the ref_action NLL
        @param configuration: the current configuration
        @param structural history: the list of structural actions performed so far
        @param ref_action  : the reference action
        @return : (NLL,correct) where NLL for the ref action and correct is true if the argmax of the distrib = ref_action
        """
        S,B,n,stack_state,lab_state,local_score = configuration
        logprobs = self.raw_action_distrib(configuration,structural_history)

        if lab_state == RNNGparser.WORD_LABEL:
            ref_prediction = self.lexicon.index(ref_action) if not self.blex else self.blex.index(ref_action)        
        elif lab_state == RNNGparser.NT_LABEL:
            ref_prediction = self.nonterminals.index(ref_action)
        else:
            ref_prediction = self.action_codes[ref_action]
            
        loss       = -dy.pick(logprobs,ref_prediction)
        loss_val   = loss.value()
        best_pred  = np.argmax(logprobs.npvalue())
        return loss_val,(best_pred==ref_prediction)

    #parsing, training, eval one sentence        
    def move_state(self,tokens,configuration,struct_history,action,score):
        """
        Applies an action to config. Use it only in greedy contexts, not with beam.
        @param tokens: the list of word tokens (as strings) of the current sentence
        @param configuration: a current config
        @param struct_history: a current struct history
        @param action: the action to execute
        @param score: the score of the action
        @return (C,H) an updated Configuration and update History resulting from applying action to prev configuration
        """
        S,B,n,stack_state,lab_state,local_score = configuration

        if lab_state == RNNGparser.WORD_LABEL:
            C = self.word_action(configuration,tokens,score)
        elif lab_state == RNNGparser.NT_LABEL:
            C = self.nonterminal_action(configuration,action,score)
        elif action == RNNGparser.CLOSE:
            C = self.close_action(configuration,score)
            struct_history.append( RNNGparser.CLOSE )
        elif action == RNNGparser.OPEN:
            C = self.open_action(configuration,score)
            struct_history.append( RNNGparser.OPEN )
        elif action == RNNGparser.SHIFT:
            C = self.shift_action(configuration,score)
            struct_history.append( RNNGparser.SHIFT )
        elif action == RNNGparser.TERMINATE:
            C = configuration
            
        return C,struct_history        
   
    def beam_parse(self,tokens,all_beam_size,lex_beam_size,kbest=1,ref_tree=None,tracker=None):
        """
        This parses a sentence with word sync beam search.
        The beam search assumes the number of structural actions between two words to be bounded 
        @param tokens: the sentence tokens
        @param all_beam_size: size of the structural beam
        @param lex_beam_size: size of the lexical beam
        @param kbest: max number of parses returned (actual number is bounded by lex_beam_size)
        @param ref_tree: if provided return an eval against ref_tree rather than a parse tree
        @param tracker: a tracker object instance, subclass of AbstractTracker.
        @return a list of either derivation, a ConsTree or some evaluation metrics. 
        """
        dy.renew_cg()

        if tracker is None:
            tracker = AbstractTracker()
        tracker.set_known_vocabulary(self.lexicon.i2words)
        tracker.next_sentence(tokens)
        
        start = BeamElement(None,'init',0)
        start.config = self.init_configuration(len(tokens))
        start.structural_history = ['init']
        
        all_beam  = [ start ]
        next_lex_beam = [ ]
        
        for idx in range(len(tokens) + 1):
            while all_beam:
                next_all_beam = []
                for elt in all_beam:
                    C = elt.config
                    _,_,_,_,lab_state,prefix_score = C
                    preds_distrib = self.predict_action_distrib(C,elt.structural_history,tokens)
                    #dispatch predicted items on relevant beams
                    if lab_state == RNNGparser.WORD_LABEL:
                        action,loc_score = preds_distrib[0]
                        next_lex_beam.append(BeamElement(elt,action,loc_score))
                    elif lab_state == RNNGparser.NT_LABEL:
                        for action,loc_score in preds_distrib:
                            next_all_beam.append(BeamElement(elt,action,loc_score))
                    else:
                        for action,loc_score in preds_distrib:
                            if action == RNNGparser.TERMINATE:
                                next_lex_beam.append(BeamElement(elt, action,loc_score))
                            else:
                                next_all_beam.append(BeamElement(elt,action,loc_score))
                #prune and exec actions
                next_all_beam.sort(key=lambda x:BeamElement.figure_of_merit(x),reverse=True)
                next_all_beam = next_all_beam[:all_beam_size]
                for elt in next_all_beam:#exec actions
                    loc_score = elt.local_score
                    action    = elt.incoming_action
                    C         = elt.prev_element.config
                    _,_,_,_,lab_state,prefix_score = C
                    if lab_state == RNNGparser.NT_LABEL:
                        elt.config = self.nonterminal_action(C,action,loc_score)
                        elt.update_history()
                    elif action == RNNGparser.CLOSE:
                        elt.config = self.close_action(C,loc_score)
                        elt.update_history(RNNGparser.CLOSE)
                    elif action == RNNGparser.OPEN:
                        elt.config = self.open_action(C,loc_score)
                        elt.update_history(RNNGparser.OPEN)
                    elif action == RNNGparser.SHIFT:
                        elt.config = self.shift_action(C,loc_score)
                        elt.update_history(RNNGparser.SHIFT)
                    else:
                        print('bug beam exec struct actions')
                all_beam = next_all_beam
                
            #Lex beam
            next_lex_beam.sort(key=lambda x:BeamElement.figure_of_merit(x),reverse=True)
            next_lex_beam = next_lex_beam[:lex_beam_size]
            for elt in next_lex_beam:
                loc_score     = elt.local_score
                action        = elt.incoming_action
                C             = elt.prev_element.config
                _,_,_,_,lab_state,prefix_score = C
                if lab_state == RNNGparser.WORD_LABEL:
                    elt.config = self.word_action(C,tokens,loc_score)
                    elt.update_history()
                    tracker.log_beam_element(elt)
                elif action == RNNGparser.TERMINATE:
                    elt.config = C
                    elt.update_history( RNNGparser.TERMINATE )
                else:
                    print('bug beam exec lex actions')
            all_beam = next_lex_beam
            next_lex_beam = [ ]
            tracker.next_word()

        #parse failure management
        if not all_beam:
            print('abort',file=sys.stderr,flush=True)
            return None

        #regular case
        results = []
        for k in range(min(kbest,len(all_beam))): #K-best results 
            #backtrace
            current    = all_beam[k]
            best_deriv = [current.incoming_action]
            while current.prev_element != None:
                current = current.prev_element
                best_deriv.append(current.incoming_action)
            best_deriv.reverse()
            pred_tree = RNNGparser.derivation2tree(best_deriv,tokens)
            pred_tree.expand_unaries() 
            if ref_tree: #returns F-score etc instead of the tree)
                results.append(ref_tree.compare(pred_tree))
            else:        #returns the tree
                results.append(pred_tree)
                
        return results
            
        
    def parse_sentence(self,tokens,get_derivation=False,ref_tree=None):
        """
        Parses a sentence greedily. if a ref_tree is provided, return Prec,Rec
        and a Fscore else returns a Constree object, the predicted
        parse tree.        
        @param tokens: a list of strings
        @param get_derivation : returns a parse derivation instead of a parse tree
        @param ref_tree: a reference PS tree
        @return a derivation, a ConsTree or some evaluation metrics
        """
        dy.renew_cg()

        C              = self.init_configuration(len(tokens))
        struct_history = ['<init>'] 
        deriv = [ ]
        pred_action    = None
        while pred_action != RNNGparser.TERMINATE :

            pred_action,score = self.predict_action_distrib(C,struct_history,tokens,max_only=True)
            deriv.append(pred_action)
            C,struct_history = self.move_state(tokens,C,struct_history,pred_action,score)

        if get_derivation:
            return deriv
        pred_tree  = RNNGparser.derivation2tree(deriv,tokens)
        if ref_tree:
            return ref_tree.compare(pred_tree)
        return pred_tree

    def train_sentence(self,ref_tree,monitor):
        """
        Trains the model on a single sentence
        @param ref_tree: a tree to train from
        @param monitor: a monitor for logging the training process
        """
        dy.renew_cg()
        tokens              = ref_tree.tokens()
        ref_derivation,_,_  = self.oracle_derivation(self.init_configuration(len(tokens)),ref_tree,tokens,['<init>'])
        step, max_step      = (0,len(ref_derivation))
        C                   = self.init_configuration(len(tokens))
        struct_history      = ['<init>'] 
        for ref_action in ref_derivation:
            NLL = self.backprop_action_distrib(C,struct_history,ref_action)
            monitor.add_NLL_datum(NLL,C)
            C,struct_history = self.move_state(tokens,C,struct_history,ref_action,-NLL)

    def eval_sentence(self,ref_tree,monitor):
        """
        Evaluates a single sentence from dev set.
        @param ref_tree: a tree to eval against
        @param monitor: a monitor for logging the eval process
        """
        dy.renew_cg()
        tokens              = ref_tree.tokens()
        ref_derivation,_,_  = self.oracle_derivation(self.init_configuration(len(tokens)),ref_tree,tokens,['<init>'])
        step, max_step      = (0,len(ref_derivation))
        C                   = self.init_configuration(len(tokens))
        struct_history      = ['<init>']
        NLL = 0 
        for ref_action in ref_derivation:
            loc_NLL,correct = self.eval_action_distrib(C,struct_history,ref_action)
            monitor.add_NLL_datum(loc_NLL,C)
            monitor.add_ACC_datum(correct,C)
            C,struct_history = self.move_state(tokens,C,struct_history,ref_action,-loc_NLL)
            NLL +=  - max(-loc_NLL,np.log(np.finfo(float).eps))  #smoothes potential zeroes
        return NLL
    
    #parsing, training etc on a full treebank
    def eval_all(self,dev_bank):
        """
        Evaluates the model on a development treebank
        """
        print('\nEval on dev...',flush=True)
        monitor =  OptimMonitor()
        D = self.dropout
        self.dropout = 0.0
        L = 0
        for tree in dev_bank:
           L += self.eval_sentence(tree,monitor)
           
        monitor.display_NLL_log(reset=True)
        monitor.display_ACC_log(reset=True)
        self.dropout = D
        return L

    def train_generative_model(self,modelname,max_epochs,train_bank,dev_bank,lex_embeddings_filename=None,cls_filename=None,learning_rate=0.001,dropout=0.3):
        """
        This trains an RNNG model on a treebank
        @param model_name: a string for the fileprefix where to store the model
        @param learning_rate: the learning rate for SGD
        @param max_epochs: the max number of epochs
        @param train_bank: a list of ConsTree
        @param dev_bank  : a list of ConsTree
        @param lex_embeddings_file: an external word embeddings filename
        @param cls_file: a path file generated by Liang's clustering package 
        @return a dynet model
        """
        self.dropout = dropout
        
        #Trees preprocessing
        for t in train_bank:
            ConsTree.strip_tags(t)
            ConsTree.close_unaries(t)

        for t in dev_bank:
            ConsTree.strip_tags(t)
            ConsTree.close_unaries(t)
            
        if cls_filename:
            print("Using clusters")
            self.blex = BrownLexicon.read_clusters(cls_filename,freq_thresh=1,UNK_SYMBOL=RNNGparser.UNKNOWN_TOKEN)
            print(self.blex.display_summary())

        #Coding
        self.code_lexicon(train_bank,self.max_vocab_size)
        self.code_nonterminals(train_bank)
        self.code_struct_actions()

        self.make_structure(lex_embeddings_filename)
        
        self.print_summary()
        print('---------------------------')
        print('num epochs          :',max_epochs)
        print('learning rate       :',learning_rate)
        print('dropout             :',self.dropout)        
        print('num training trees  :',len(train_bank),flush=True)
        
        #training
        self.trainer = dy.AdamTrainer(self.model,alpha=learning_rate)
        best_model_loss = np.inf     #stores the best model on dev
        monitor =  OptimMonitor()
        for e in range(max_epochs):
            print('\n--------------------------\nEpoch %d'%(e,),flush=True)
            for idx,tree in enumerate(train_bank):
                sys.stdout.write('\rtree #%d/%d        '%(idx,len(train_bank)))
                sys.stdout.flush()
                self.train_sentence(tree,monitor)
                
                if (idx+1) % 1000 == 0:
                    monitor.display_NLL_log(tree_idx=idx)
                    print()
                     
            monitor.display_NLL_log(reset=True)            
            devloss = self.eval_all(dev_bank)
            #shuffle(train_bank)
            if devloss <= best_model_loss :
                best_model_loss=devloss
                print(" => saving model",devloss)
                self.save_model(modelname)
                monitor.save_loss_curves(modelname+'.learningcurves.csv')
        print()
        self.save_model(modelname+'.final')
        monitor.save_loss_curves(modelname+'.learningcurves.csv')
        self.dropout = 0.0  #prevents dropout to be applied at decoding
            
    #I/O etc.
    def print_summary(self):
        """
        Prints a summary of the parser structure
        """

        lexA =   self.blex.size() if self.blex else self.lexicon.size()
        print('Num Lexical actions     :',lexA,flush=True)
        print('Num NT actions          :',self.nonterminals.size(),flush=True)
        print('Num struct actions      :',len(self.actions),flush=True)
        print('Lexicon size            :',self.lexicon.size(),flush=True)
        print('Stack embedding size    :',self.stack_embedding_size,flush=True)
        print('Stack hidden size       :',self.stack_hidden_size,flush=True)

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
                
    @staticmethod
    def load_model(model_name):
        """
        Loads the whole shebang and returns a parser.
        """            
        struct = json.loads(open(model_name+'.json').read())
        parser              = RNNGparser(stack_embedding_size = struct['stack_embedding_size'],stack_memory_size=struct['stack_hidden_size'])
        parser.lexicon      = SymbolLexicon.load(model_name+'.lex')
        parser.nonterminals = SymbolLexicon.load(model_name+'.nt')
        try:
            parser.blex = BrownLexicon.load_clusters(model_name+'.cls')
        except FileNotFoundError:
            print('No clusters found',file=sys.stderr)
            parser.blex = None
        parser.code_struct_actions()
        parser.make_structure()
        parser.model.populate(model_name+".prm")
        return parser
    
    def save_model(self,model_name):
        """
        Saves the whole shebang.
        """
        jfile = open(model_name+'.json','w')
        jfile.write(json.dumps({'stack_embedding_size':self.stack_embedding_size,'stack_hidden_size':self.stack_hidden_size}))
        jfile.close()
        
        self.model.save(model_name+'.prm')
        self.lexicon.save(model_name+'.lex')
        self.nonterminals.save(model_name+'.nt')
        if self.blex:
            self.blex.save_clusters(model_name+'.cls')
        
  
if __name__ == '__main__':
    
    warnings.simplefilter("ignore")
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:o:d:r:m:b:L:S:e:c:p:k:")
    except getopt.GetoptError:
        print ('rnng.py -t <inputfile> -d <inputfile> -r <inputfile> -o <outputfile> -m <model_file>')
        sys.exit(0)

    train_file     = ''
    dev_file       = ''
    model_name     = ''
    brown_file     = None
    embedding_file = None 
    raw_file     = ''
    predict_file = ''
    lex_beam     = 10  #40
    struct_beam  = 100 #400
    kbest        = 1
    config_file  = 'defaultconfig.prm'
    check_oracle = False
    
    for opt, arg in opts:
        if opt in ['-h','--help']:
            print ('rnng.py -t <inputfile> -d <inputfile> -r <inputfile> -m <model_name>')
            sys.exit(0)
        elif opt in ['-t','--train']:
            train_file = arg
        elif opt in ['-d','--dev']:
            dev_file = arg
        elif opt in ['-r','--raw']:
            raw_file = arg
        elif opt in ['-m','--model']:
            model_name = arg
        elif opt in ['-L','--lex-beam']:
            lex_beam = int(arg)
        elif opt in ['-S','--struct-beam']:
            struct_beam = int(arg)
        elif opt in ['-b','--brown']:
            brown_file = arg
        elif opt in ['-e','--embedding']:
            embedding_file = arg
        elif opt in ['-c','--configfile']:
            config_file = arg
        elif opt in ['-p','--predict']:
            predict_file = arg
        elif opt in ['-k','--kbest']:
            kbest = int(arg)

    if train_file and model_name: #train
        read_config(config_file)
        train_treebank = []
        train_stream   = open(train_file)
        for line in train_stream:
            train_treebank.append(ConsTree.read_tree(line))
        train_stream.close()
        
        dev_treebank = []
        if dev_file:
            dev_stream   = open(train_file)
            for line in dev_stream:
                dev_treebank.append(ConsTree.read_tree(line))
            dev_stream.close()
            
        p = RNNGparser(max_vocabulary_size=TrainingParams.LEX_MAX_SIZE,\
                        stack_embedding_size=StructParams.STACK_EMB_SIZE,\
                        stack_memory_size=StructParams.STACK_HIDDEN_SIZE)
        p.train_generative_model(model_name,\
                                 TrainingParams.NUM_EPOCHS,\
                                 train_treebank,\
                                 dev_treebank,\
                                 learning_rate=TrainingParams.LEARNING_RATE,\
                                 dropout=TrainingParams.DROPOUT,\
                                 cls_filename=brown_file,\
                                 lex_embeddings_filename=embedding_file)
        
    #runs a test on raw text   
    elif model_name and raw_file:
        p = RNNGparser.load_model(model_name)
        test_istream  = open(raw_file)
        out_name = '.'.join(raw_file.split('.')[:-1]+['mrg'])
        test_ostream  = open(model_name+'-'+out_name,'w') 
        for line in test_istream:
            result = p.beam_parse(line.split(),all_beam_size=struct_beam,lex_beam_size=lex_beam,kbest=kbest)
            for elt in result:
                print(elt,file=test_ostream,flush=True)
        test_istream.close()
        test_ostream.close()

    #runs a test on pos tagged text (! predict file is a treebank) 
    elif model_name and predict_file:
        p = RNNGparser.load_model(model_name)
        test_istream  = open(predict_file)
        out_name = '.'.join(predict_file.split('.')[:-1]+['pred.mrg'])
        test_ostream  = open(model_name+'-'+out_name,'w') 
        parse_tracker = DefaultTracker(model_name+'.csv')
        for line in test_istream:
            tree = ConsTree.read_tree(line)
            wordsXtags = tree.pos_tags()
            words = [elt.get_child().label for elt in wordsXtags]
            tags  = [elt.label for elt in wordsXtags]
            result = p.beam_parse(words,all_beam_size=struct_beam,lex_beam_size=lex_beam,tracker=parse_tracker,kbest=kbest)
            for elt in result:
                if elt:
                    elt.add_gold_tags(tags)
                    print(elt,file=test_ostream,flush=True)
            #print('*',flush=True,file=sys.stderr)
            
        test_istream.close()
        test_ostream.close()
        parse_tracker.save_table()
        print(parse_tracker.get_corpus_NLL(ppl=True))
        
    #despaired debugging
    elif not model_name:
        if train_file:#takes the 20 first sentences
            train_treebank = []
            train_stream   = open(train_file)
            idx  = 0
            for line in train_stream:
                if idx >= 0:
                    train_treebank.append(ConsTree.read_tree(line))
                idx += 1
                if idx >= 20:
                    break
            train_stream.close()
            StructParams.STACK_EMB_SIZE = 100
            StructParams.STACK_HIDDEN_SIZE = 100
            TrainingParams.DROPOUT = 0.1
            TrainingParams.NUM_EPOCHS = 20
        else:
            t  = ConsTree.read_tree('(S (NP Le chat ) (VP mange  (NP la souris)))')
            t2 = ConsTree.read_tree('(S (NP Le chat ) (VP voit  (NP le chien) (PP sur (NP le paillasson))))')
            t3 = ConsTree.read_tree('(S (NP La souris (Srel qui (VP dort (PP sur (NP le paillasson))))) (VP sera mangée (PP par (NP le chat ))))')
            train_treebank = [t,t2,t3]
        
        p = RNNGparser(max_vocabulary_size=TrainingParams.LEX_MAX_SIZE,\
                        stack_embedding_size=StructParams.STACK_EMB_SIZE,\
                        stack_memory_size=StructParams.STACK_HIDDEN_SIZE)
        p.train_generative_model('none',TrainingParams.NUM_EPOCHS,train_treebank,[t.copy() for t in train_treebank],learning_rate=TrainingParams.LEARNING_RATE,dropout=TrainingParams.DROPOUT,cls_filename=brown_file,lex_embeddings_filename=embedding_file)
        dtracker = DefaultTracker('cog_stats.csv')
        for t in train_treebank:
            #print(p.parse_sentence(t.tokens()))
            wordsXtags = t.pos_tags()
            words = [elt.get_child().label for elt in wordsXtags]
            tags  = [elt.label for elt in wordsXtags]
            results = p.beam_parse(words,all_beam_size=struct_beam,lex_beam_size=lex_beam,kbest=kbest,tracker=dtracker)
            for elt in results:
                if elt:
                    elt.add_gold_tags(tags)
                    print("%s"%(str(elt)),file=test_ostream,flush=True)
            #print('\n'.join(["%s %f"%(str(r),t.compare(r)[2]) for r in results]))
            #print(p.beam_parse(t.tokens(),all_beam_size=struct_beam,lex_beam_size=lex_beam,tracker=dtracker))
        dtracker.save_table()
        print()
        p.save_model('none')
        p.load_model('none')
        dtracker = DefaultTracker('cog_stats.csv')
        for t in train_treebank:
            #print(p.parse_sentence(t.tokens()))         
            results = p.beam_parse(t.tokens(),all_beam_size=struct_beam,lex_beam_size=lex_beam,tracker=dtracker)
            print('\n'.join(["%s %f"%(str(r),t.compare(r)[2]) for r in results]))            
        dtracker.save_table()
        
