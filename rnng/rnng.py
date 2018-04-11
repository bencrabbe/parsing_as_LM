import numpy as np
import dynet as dy
import getopt
import json
from collections import Counter
from constree import *
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
                 hidden_size = 50,
                 stack_embedding_size=50,
                 stack_memory_size=50):
        """
        @param max_vocabulary_size     : max number of words in the vocab
        @param stack_embedding_size    : size of stack lstm input 
        @param stack_memory_size       : size of the stack and tree lstm hidden layers
        @param hidden_size             : size of the output hidden layer
        """
        self.max_vocab_size       = max_vocabulary_size
        self.stack_embedding_size = stack_embedding_size
        self.stack_hidden_size    = stack_memory_size
        self.hidden_size          = hidden_size
        self.dropout = 0.0

    def oracle_derivation(self,ref_tree,root=True):
        """
        Returns an oracle derivation given a reference tree
        @param ref_tree: a ConsTree
        @return a list of (action, configuration) couples (= a derivation)
        """
        if ref_tree.is_leaf():
            return [RNNGparser.SHIFT,ref_tree.label]
        else:
            first_child = ref_tree.children[0]
            derivation = self.oracle_derivation(first_child,root=False)
            derivation.extend([RNNGparser.OPEN,ref_tree.label])
            for child in ref_tree.children[1:]: 
                derivation.extend(self.oracle_derivation(child,root=False))
            derivation.append(RNNGparser.CLOSE)
        if root:
            derivation.append(RNNGparser.TERMINATE)
        return derivation

    @staticmethod
    def derivation2tree(derivation):
        """
        Transforms a derivation back to a tree
        @param derivation: a derivation to use for generating a tree
        """
        stack = []
        prev_action = None
        for action in derivation:
            if prev_action == RNNGparser.SHIFT:
                lex = ConsTree(action)
                stack.append((lex,False))
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
        print('Full lexicon size (prior to capping):',len(lexicon))
        lexicon = set([word for word,count in lexicon.most_common(max_vocab_size-2)])
        lexicon.add(RNNGparser.UNKNOWN_TOKEN)
        lexicon.add(RNNGparser.START_TOKEN)
        self.rev_word_codes = list(lexicon)
        self.lexicon_size   = len(lexicon)
        self.word_codes     = dict([(s,idx) for (idx,s) in enumerate(self.rev_word_codes)])

    def lex_lookup(self,token):
        """
        Performs lookup and backs off unk words to the unk token
        @param token : the token to code
        @return : identity for in-vocab tokens and unk word string for OOV tokens
        """
        return token if token in self.word_codes else RNNGparser.UNKNOWN_TOKEN
        
    def code_nonterminals(self,treebank):
        """
        Extracts the nonterminals from a treebank.
        @param treebank: the treebank where to extract the data from
        """
        self.nonterminals = set([])
        for tree in treebank:
            self.nonterminals.update(tree.collect_nonterminals())
        self.nonterminals       = list(self.nonterminals)
        self.nonterminals_codes = dict([(sym,idx) for (idx,sym) in enumerate(self.nonterminals)])
        return self.nonterminals

    def code_struct_actions(self):
        """
        Codes the structural actions on integers
        """
        self.actions         = [RNNGparser.SHIFT,RNNGparser.OPEN,RNNGparser.CLOSE,RNNGparser.TERMINATE]
        self.action_codes    = dict([(sym,idx) for (idx,sym) in enumerate(self.actions)])
        self.open_mask       = np.log([True,False,True,True]) #we assume scores are log probs
        self.shift_mask      = np.log([False,True,True,True])
        self.close_mask      = np.log([True,True,False,True])
        self.terminate_mask  = np.log([True,True,True,False])

        
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

        word_idx = self.word_codes[RNNGparser.START_TOKEN]
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
        @param sentence: the list of words of the sentence as a list of word idxes
        @param local_score: the local score of the action (logprob)
        @return a configuration resulting from shifting the next word into the stack 
        """
        S,B,n,stack_state,lab_state,score = configuration
        word_idx = sentence[B[0]]
        if self.dropout > 0.0:
            word_embedding = dy.dropout(self.lex_embedding_matrix[word_idx],self.dropout)
        else: 
            word_embedding = self.lex_embedding_matrix[word_idx]
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

        nt_idx = self.nonterminals_codes[X]

        if self.dropout > 0.0:
            nonterminal_embedding = dy.dropout(self.nt_embedding_matrix[nt_idx],self.dropout)
        else:
            nonterminal_embedding = self.nt_embedding_matrix[nt_idx]
            
        stack_state = stack_state.add_input(nonterminal_embedding)
        return (S + [StackSymbol(X,StackSymbol.PREDICTED,nonterminal_embedding)],B,n+1,stack_state,RNNGparser.NO_LABEL,score+local_score)

    
    def close_action(self,configuration,local_score):
        """
        That's the RNNG CLOSE action.
        @param configuration : a configuration tuple
        @param local_score: the local score of the action (logprob)
        @return a configuration resulting from closing the current constituent
        """
        S,B,n,stack_state,lab_state,score = configuration
        assert( n > 0 )
        #finds the closest predicted constituent in the stack and backtracks the stack lstm.
        midx = -1
        for idx,symbol in enumerate(reversed(S)):
            if symbol.status == StackSymbol.PREDICTED:
                midx = idx+2
                break
            else:
                stack_state = stack_state.prev()
        stack_state = stack_state.prev()
        root_symbol = S[-midx+1].copy()
        root_symbol.complete()
        children    = [S[-midx]]+S[-midx+2:]
            
        #compute the tree embedding with the tree_rnn
        nt_idx = self.nonterminals_codes[root_symbol.symbol]
        if self.dropout > 0.0:
            NT_embedding = self.nt_embedding_matrix[nt_idx]
        else:
            NT_embedding = dy.dropout(self.nt_embedding_matrix[nt_idx],self.dropout)
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

        if self.dropout > 0.0:
            tree_embedding = dy.dropout(dy.tanh(W * x),self.dropout)
        else:
            tree_embedding = dy.tanh(W * x)
        
        return (S[:-midx]+[root_symbol],B,n-1,stack_state.add_input(tree_embedding),RNNGparser.NO_LABEL,score+local_score)

    def structural_action_mask(self,configuration,last_structural_action,sentence):
        """ 
        This returns a mask stating which abstract actions are possible for next round
        @param configuration: the current configuration
        @param last_structural_action: the last action performed.
        @param sentence     : a list of strings, the tokens
        @return a mask for the possible next actions
        """
        #Assumes masking log probs
        MASK = np.log([True] * len(self.actions))
        S,B,n,stack_state,lab_state,local_score = configuration

        if not B or not S or last_structural_action == RNNGparser.OPEN:
            MASK += self.open_mask
        if B or n > 0 or len(S) > 1:
            MASK += self.terminate_mask
        if not B:
            MASK += self.shift_mask
        if not S or last_structural_action == RNNGparser.OPEN or n == 0:
            MASK += self.close_mask
        return MASK

    
    def predict_action_distrib(self,configuration,last_structural_action,sentence,max_only=False):
        """
        This predicts the next action distribution with the classifier and constrains it with the classifier structural rules
        @param configuration: the current configuration
        @param last_structural_action  : the last structural action performed by this parser
        @param sentence : a list of string tokens
        @param max_only : returns only the couple (action,logprob) with highest score
        @return a list of (action,logprob) legal at that state
        """        
        S,B,n,stack_state,lab_state,local_score = configuration

        if lab_state == RNNGparser.WORD_LABEL: #generate wordform action
            next_word = sentence[B[0]]
            W = dy.parameter(self.lex_out)
            b = dy.parameter(self.lex_bias)
            logprobs = dy.log_softmax(W * dy.tanh(stack_state.output()) + b).npvalue()
            score = np.maximum(logprobs[self.word_codes[next_word]],np.log(np.finfo(float).eps))
            if max_only:
                return (next_word,score)
            else:
                return [(next_word,score)]
            
        elif lab_state == RNNGparser.NT_LABEL: #label NT action
            W = dy.parameter(self.nt_out)
            b = dy.parameter(self.nt_bias)
            logprobs = dy.log_softmax(W * dy.tanh(stack_state.output()) + b).npvalue()
            if max_only:
                idx = np.argmax(logprobs)
                return (self.nonterminals[idx],logprobs[idx])
            else:
                return list(zip(self.nonterminals,logprobs))
        
        else: #lab_state == RNNGparser.NO_LABEL perform a structural action
            W = dy.parameter(self.struct_out)
            b = dy.parameter(self.struct_bias)
            logprobs = dy.log_softmax(W * dy.tanh(stack_state.output()) + b).npvalue()
            #constraint + underflow prevention
            logprobs = np.maximum(logprobs,np.log(np.finfo(float).eps)) + self.structural_action_mask(configuration,last_structural_action,sentence)
            if max_only:
                idx = np.argmax(logprobs)
                #TODO here:if logprob == -inf raise parse failure 
                return (self.actions[idx],logprobs[idx])
            else:
                return [(act,logp) for act,logp in zip(self.actions,logprobs) if logp > -np.inf]
        
    def train_one(self,configuration,ref_action):
        """
        This performs a forward, backward and update pass on the network for this action.
        @param configuration: the current configuration
        @param ref_action  : the reference action
        @return (the loss for this action,a boolean indicating if the prediction argmax is correct or not)
        """
        S,B,n,stack_state,lab_state,local_score = configuration

        if lab_state == RNNGparser.WORD_LABEL:
            W   = dy.parameter(self.lex_out)
            b   = dy.parameter(self.lex_bias)
            correct_prediction = self.word_codes[ref_action]
        elif lab_state == RNNGparser.NT_LABEL:
            W   = dy.parameter(self.nt_out)
            b   = dy.parameter(self.nt_bias)
            correct_prediction = self.nonterminals_codes[ref_action]
        else:
            W   = dy.parameter(self.struct_out)
            b   = dy.parameter(self.struct_bias)
            correct_prediction = self.action_codes[ref_action]
            
        log_probs = dy.log_softmax( (W * dy.dropout(dy.tanh(stack_state.output()),self.dropout)) + b)
        best_prediction = np.argmax(log_probs.npvalue())
        iscorrect = (correct_prediction == best_prediction)
        loss       = dy.pick(-log_probs,correct_prediction)
        loss_val   = loss.value()
        loss.backward()
        self.trainer.update()
        return loss_val,iscorrect

    
    def beam_parse(self,tokens,all_beam_size,lex_beam_size,ref_tree=None):
        """
        This parses a sentence with word sync beam search.
        The beam search assumes the number of structural actions between two words to be bounded 
        @param tokens: the sentence tokens
        @param ref_tree: if provided return an eval against ref_tree rather than a parse tree
        @return a derivation, a ConsTree or some evaluation metrics
        """
        class BeamElement:
            
            #representation used for lazy delayed exec of actions in the beam
            def __init__(self,prev_item,last_structural_action,current_action,local_score):
                self.prev_element           = prev_item               #prev beam item (history)
                self.last_structural_action = last_structural_action  #scheduling info
                self.incoming_action        = current_action
                self.config                 = None           
                self.local_score            = local_score
            @staticmethod
            def figure_of_merit(elt):
                #provides a score for ranking the elements in the beam
                #could add derivation length for further normalization (?)
                _,_,_,_,lab_state,prefix_score = self.prev_element.config
                return self.local_score + prefix_score
                
        dy.renew_cg()
        tokens    = [self.lex_lookup(t) for t in tokens  ]
        tok_codes = [self.word_codes[t] for t in tokens  ]    
        start = BeamElement(None,'init','init',0)
        start.config = self.init_configuration(len(tokens))

        all_beam  = [ start ]
        next_lex_beam = [ ]
        
        for idx in range(len(tokens) + 1):
            while all_beam:
                next_all_beam = []
                for elt in all_beam:
                    C = elt.config
                    _,_,_,_,lab_state,prefix_score = C
                    prev_s_action = elt.last_structural_action
                    preds_distrib = self.predict_action_distrib(C,prev_s_action,tokens)
                    #dispatch predicted items on relevant beams
                    if lab_state == RNNGparser.WORD_LABEL: 
                        action,loc_score = preds_distrib[0]
                        next_lex_beam.append(BeamElement(elt,prev_s_action,action,loc_score))
                    elif lab_state == RNNGparser.NT_LABEL:
                        for action,loc_score in preds_distrib:
                            next_all_beam.append(BeamElement(elt,prev_s_action,action,loc_score))
                    else:
                        for action,loc_score in preds_distrib:
                            print('struct',action)
                            if action == RNNGparser.TERMINATE:
                                next_lex_beam.append(BeamElement(elt,prev_s_action, action,loc_score))
                            else:
                                next_all_beam.append(BeamElement(elt,prev_s_action,action,loc_score))
                #prune and exec actions
                next_all_beam.sort(key=lambda x:BeamElement.figure_of_merit(x),reverse=True)
                next_all_beam = next_all_beam[:all_beam_size]
                for elt in next_all_beam:#exec actions
                    loc_score = elt.local_score
                    action    = elt.incoming_action
                    C         = elt.prev_element.config
                    _,_,_,_,lab_state,prefix_score = C
                    prev_s_action = elt.last_structural_action
                    if lab_state == RNNGparser.NT_LABEL:
                        elt.config = self.nonterminal_action(C,action,loc_score)
                    elif pred_action == RNNGparser.CLOSE:
                        elt.config = self.close_action(C,loc_score)
                        elt.last_struct_action = RNNGparser.CLOSE
                    elif pred_action == RNNGparser.OPEN:
                        elt.config = self.open_action(C,loc_score)
                        elt.last_struct_action = RNNGparser.OPEN
                    elif pred_action == RNNGparser.SHIFT:
                        elt.config = self.shift_action(C,loc_score)
                        elt.last_struct_action = RNNGparser.SHIFT
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
                prev_s_action = elt.last_structural_action
                if lab_state == RNNGparser.WORD_LABEL:
                    elt.config = self.word_action(C,tok_codes,loc_score)
                if act == RNNGparser.TERMINATE:
                    elt.config = C
                    elt.last_struct_action = RNNGparser.TERMINATE
            all_beam = next_lex_beam
            next_lex_beam = [ ]
        #backtrace
        current    = all_beam[0]
        best_deriv = [current.pred_action]
        while current.prev != None:
            current = current.prev
            best_deriv.append(current.pred_action)
        best_deriv.reverse()

        pred_tree = RNNGparser.derivation2tree(best_deriv) 
        if ref_tree:
            return ref_tree.compare(pred_tree)
        return pred_tree

    def parse_sentence(self,tokens,get_derivation=False,ref_tree=None):
        """
        Parses a sentence. if a ref_tree is provided, return Prec,Rec
        and a Fscore else returns a Constree object, the predicted
        parse tree.        
        @param tokens: a list of strings
        @param get_derivation : returns a parse derivation instead of a parse tree
        @param ref_tree: a reference PS tree
        @return a derivation, a ConsTree or some evaluation metrics
        """
        dy.renew_cg()
        tokens    = [self.lex_lookup(t) for t in tokens  ]
        tok_codes = [self.word_codes[t] for t in tokens  ]
        C         = self.init_configuration(len(tokens))

        last_struct_action = 'init'
         
        S,B,n,stackS,lab_state,score = C
        deriv = [ ]
        while True:
            (pred_action,score) = self.predict_action_distrib(C,last_struct_action,tokens,max_only=True)
            deriv.append(pred_action)
            if lab_state == RNNGparser.WORD_LABEL:
                C = self.word_action(C,tok_codes,score)
            elif lab_state == RNNGparser.NT_LABEL:
                C = self.nonterminal_action(C,pred_action,score)
            elif pred_action == RNNGparser.CLOSE:
                C = self.close_action(C,score)
                last_struct_action = RNNGparser.CLOSE
            elif pred_action == RNNGparser.OPEN:
                C = self.open_action(C,score)
                last_struct_action = RNNGparser.OPEN
            elif pred_action == RNNGparser.SHIFT:
                C = self.shift_action(C,score)
                last_struct_action = RNNGparser.SHIFT
            elif pred_action == RNNGparser.TERMINATE:
                break
            
            S,B,n,stackS,lab_state,score = C
                        
        if get_derivation:
            return deriv
        pred_tree  = RNNGparser.derivation2tree(deriv)
        if ref_tree:
            return ref_tree.compare(pred_tree)
        return pred_tree

    def print_summary(self):
        """
        Prints the summary of the parser setup
        """
        print('Lexicon size            :',len(self.rev_word_codes),flush=True)
        print('Non terminals size      :',len(self.nonterminals),flush=True)
        print('Number of actions       :',len(self.actions),flush=True)
        print('Outer hidden layer size :',self.hidden_size,flush=True)
        print('Stack embedding size    :',self.stack_embedding_size,flush=True)
        print('Stack hidden size       :',self.stack_hidden_size,flush=True)

    def make_structure(self):
        """
        Allocates the network structure
        """
        lexicon_size = len(self.rev_word_codes)
        actions_size = len(self.actions)
        nt_size      = len(self.nonterminals)

        #Model structure
        self.model                 = dy.ParameterCollection()
        
        #top level task predictions
        self.struct_out             = self.model.add_parameters((actions_size,self.hidden_size),init='glorot')          #struct action output layer
        self.struct_bias            = self.model.add_parameters((actions_size),init='glorot')

        self.lex_out                = self.model.add_parameters((lexicon_size,self.hidden_size),init='glorot')          #lex action output layer
        self.lex_bias               = self.model.add_parameters((lexicon_size),init='glorot')

        self.nt_out                 = self.model.add_parameters((nt_size,self.hidden_size),init='glorot')               #nonterminal action output layer
        self.nt_bias                = self.model.add_parameters((nt_size),init='glorot')

        
        #embeddings
        self.lex_embedding_matrix  = self.model.add_lookup_parameters((lexicon_size,self.stack_embedding_size),init='glorot')       # symbols embeddings
        self.nt_embedding_matrix   = self.model.add_lookup_parameters((nt_size,self.stack_embedding_size),init='glorot')
        #stack rnn 
        self.stack_rnn             = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)        # main stack rnn
        #tree rnn
        self.fwd_tree_rnn          = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)        # bi-rnn for tree embeddings
        self.bwd_tree_rnn          = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)
        self.tree_rnn_out          = self.model.add_parameters((self.stack_embedding_size,self.stack_hidden_size*2),init='glorot')       # out layer merging the tree bi-rnn output

    @staticmethod
    def load_model(model_name):
        """
        Loads the whole shebang and returns a parser.
        """
        struct = json.loads(open(model_name+'.json').read())
        parser = RNNGparser(max_vocabulary_size=struct['max_vocabulary_size'],
                 hidden_size = struct['hidden_size'],
                 stack_embedding_size = struct['stack_embedding_size'],
                 stack_memory_size= struct['stack_hidden_size'])
        parser.rev_word_codes     = struct['rev_word_codes']
        parser.nonterminals       = struct['nonterminals']
        parser.nonterminals_codes = dict([(sym,idx) for (idx,sym) in enumerate(parser.nonterminals)])
        parser.word_codes         = dict([(s,idx) for (idx,s) in enumerate(parser.rev_word_codes)])
        parser.code_actions()
        parser.make_structure()
        parser.model.populate(model_name+".prm")
        return parser

    
    def save_model(self,model_name):
        """
        Saves the whole shebang.
        """
        jfile = open(model_name+'.json','w')
        jfile.write(json.dumps({'max_vocabulary_size':self.max_vocab_size,\
                                'stack_embedding_size':self.stack_embedding_size,\
                                'stack_hidden_size':self.stack_hidden_size,\
                                'hidden_size':self.hidden_size,
                                'nonterminals': self.nonterminals,
                                'rev_word_codes':self.rev_word_codes}))
        self.model.save(model_name+'.prm')

        
    def eval_trees(self,ref_treebank,all_beam_size,lex_beam_size):
        """
        Returns a pseudo f-score of the model against ref_treebank
        with all_beam_size and lex_beam_size.
        This F-score is not equivalent to evalb f-score and should be regarded as indicative only.
        @param ref_treebank : the treebank to evaluate on
        @return Prec,Recall,F-score
        """
        P,R,F = 0.0,0.0,0.0
        N     = len(ref_treebank) 
        for tree in ref_treebank:
            p,r,f = self.beam_parse(tree.tokens(),all_beam_size,lex_beam_size,ref_tree=tree)
            P+=p
            R+=r
            F+=f
        return P/N,R/N,F/N
                    
    def train_generative_model(self,max_epochs,train_bank,dev_bank,lex_embeddings_file=None,learning_rate=0.001,dropout=0.3):
        """
        This trains an RNNG model on a treebank
        @param learning_rate: the learning rate for SGD
        @param max_epochs: the max number of epochs
        @param train_bank: a list of ConsTree
        @param dev_bank  : a list of ConsTree
        @param lex_embeddings_file: an external word embeddings filename
        @return a dynet model
        """
        self.dropout = dropout
        
        #Coding
        self.code_lexicon(train_bank,self.max_vocab_size)
        self.code_nonterminals(train_bank)
        self.code_struct_actions()

        self.print_summary()
        print('---------------------------')
        print('num epochs          :',max_epochs)
        print('learning rate       :',learning_rate)
        print('dropout             :',self.dropout)        
        print('num training trees  :',len(train_bank),flush=True)

        self.make_structure()

        
        lexicon = set(self.rev_word_codes)
        for t in train_bank:
            t.normalize_OOV(lexicon,RNNGparser.UNKNOWN_TOKEN)
        for t in dev_bank:
            t.normalize_OOV(lexicon,RNNGparser.UNKNOWN_TOKEN)
            
        #training
        self.trainer = dy.AdamTrainer(self.model,alpha=learning_rate)

        #Monitoring loss & accurracy
        class OptimMonitor:
            def __init__(self,step_size=1000):
                self.step_size = step_size
                self.N = 0
                self.reset_all()
            def reset_all(self):
                if self.N > 0:
                    sys.stdout.write("\nEpoch %d, Mean Loss : %.5f\n"%(e,self.loss/self.N))
                    sys.stdout.flush()
                self.reset_loss_counts()
                self.reset_acc_counts()
            def reset_loss_counts(self):
                self.loss = 0
                self.N    = 0
            def reset_acc_counts(self):
                self.acc_sum = 0
            def add_datum(self,datum_loss,datum_correct):
                self.loss  += datum_loss
                self.N     +=1
                self.acc_sum+=datum_correct
                if self.N % self.step_size == 0:
                    sys.stdout.write("\r    Mean acc (%d): %.5f"%(self.step_size,self.acc_sum/self.step_size))
                    self.reset_acc_counts()
                
        for e in range(max_epochs):
            monitor =  OptimMonitor()
            for tree in train_bank:
                dy.renew_cg()
                ref_derivation  = self.oracle_derivation(tree)
                tok_codes = [self.word_codes[t] for t in tree.tokens()]   
                step, max_step  = (0,len(ref_derivation))
                C               = self.init_configuration(len(tok_codes))
                while step < max_step:
                    ref_action = ref_derivation[step]
                    loc_loss,correct = self.train_one(C,ref_action)
                    monitor.add_datum(loc_loss,correct)

                    S,B,n,stackS,lab_state,score = C
                    if lab_state == RNNGparser.WORD_LABEL:
                        C = self.word_action(C,tok_codes,0)
                    elif lab_state == RNNGparser.NT_LABEL:
                        C = self.nonterminal_action(C,ref_action,0)
                    elif ref_action == RNNGparser.CLOSE:
                        C = self.close_action(C,0)
                    elif ref_action == RNNGparser.OPEN:
                        C = self.open_action(C,0)
                    elif ref_action == RNNGparser.SHIFT:
                        C = self.shift_action(C,0)
                    elif ref_action == RNNGparser.TERMINATE:
                        break
                    step+=1
                    
            monitor.reset_all()
        print()
        self.dropout = 0.0  #prevents dropout to be applied at decoding

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ht:o:d:r:m:")
    except getopt.GetoptError:
        print ('rnng.py -t <inputfile> -d <inputfile> -r <inputfile> -o <outputfile> -m <model_file>')
        sys.exit(0)

    train_file = ''
    out_file   = ''
    model_name = ''
    raw_file   = ''
    lex_beam    = 8
    struct_beam = 64
    
    for opt, arg in opts:
        if opt in ['-h','--help']:
            print ('rnng.py -t <inputfile> -d <inputfile> -r <inputfile> -o <outputfile> -m <model_name>')
            sys.exit(0)
        elif opt in ['-t','--train']:
            train_file = arg
        elif opt in ['-d','--dev']:
            dev_file = arg
        elif opt in ['-r','--raw']:
            raw_file = arg
        elif opt in ['-m','--model']:
            model_name = arg
        elif opt in ['-o','--output']:
            out_file = arg
        elif opt in ['--lex-beam']:
            lex_beam = int(arg)
        elif opt in ['--struct-beam']:
            struct_beam = int(arg)
            
    train_treebank = []

    if train_file and model_name:
        train_treebank = []
        train_stream   = open(train_file)
        for line in train_stream:
            train_treebank.append(ConsTree.read_tree(line))
        p = RNNGparser(max_vocabulary_size=TrainingParams.LEX_MAX_SIZE,\
                        hidden_size=StructParams.OUTER_HIDDEN_SIZE,\
                        stack_embedding_size=StructParams.STACK_EMB_SIZE,\
                        stack_memory_size=StructParams.STACK_HIDDEN_SIZE)
        p.train_generative_model(TrainingParams.NUM_EPOCHS,train_treebank,[],learning_rate=TrainingParams.LEARNING_RATE,dropout=TrainingParams.DROPOUT)
        p.save_model(model_name)
        train_stream.close()
        #runs a test on train data
        for t in train_treebank[:100]:
            print(p.beam_parse(t.tokens(),all_beam_size=64,lex_beam_size=8))
        
    #runs a test    
    if model_name and raw_file:
        p = RNNGparser.load_model(model_name)
        test_stream = open(raw_file)
        for line in test_stream:
            #print(p.parse_sentence(line.split(),ref_tree=None))
            print(p.beam_parse(line.split(),all_beam_size=struct_beam,lex_beam_size=lex_beam))
        test_stream.close()

        
    #despaired debugging
    if not model_name:
        t  = ConsTree.read_tree('(S (NP Le chat ) (VP mange  (NP la souris)))')
        t2 = ConsTree.read_tree('(S (NP Le chat ) (VP voit  (NP le chien) (PP sur (NP le paillasson))))')
        t3 = ConsTree.read_tree('(S (NP La souris (Srel qui (VP dort (PP sur (NP le paillasson))))) (VP sera mangée (PP par (NP le chat ))))')
        train_treebank = [t,t2,t3]
        
        p = RNNGparser(max_vocabulary_size=TrainingParams.LEX_MAX_SIZE,\
                        hidden_size=StructParams.OUTER_HIDDEN_SIZE,\
                        stack_embedding_size=StructParams.STACK_EMB_SIZE,\
                        stack_memory_size=StructParams.STACK_HIDDEN_SIZE)
        p.train_generative_model(TrainingParams.NUM_EPOCHS,train_treebank,[],learning_rate=TrainingParams.LEARNING_RATE,dropout=TrainingParams.DROPOUT)
        for t in train_treebank:
            print(p.parse_sentence(t.tokens()))         
            print(p.beam_parse(t.tokens(),all_beam_size=struct_beam,lex_beam_size=lex_beam))
