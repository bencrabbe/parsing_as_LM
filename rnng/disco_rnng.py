#!/usr/bin/env python

import dynet as dy
from collections import namedtuple

from lexicons  import *
from discotree import *
from proc_monitors import *
from rnng_params import *

class StackSymbol:
    """
    A convenience class for symbols on the stack.
    """
    def __init__(self,symbol,embedding,predicted=False,sym_range=None):
        """
        Args:
           symbol           (string): a non terminal or a word
           embedding (dy.expression): a dynet expression being the embedding of the subtree dominated by this symbol (or word)
        KwArgs:
           predicted          (bool): predicted or completed
           sym_range           (set): the yield of this symbol as a set of integers
        """
        self.symbol,self.embedding,self.predicted = symbol,embedding,predicted
        self.range = sym_range
        self.has_to_move                              = False

    def copy(self): 
        """
        Returns:
           A StackSymbol object, a copy of this symobl
        """
        s = StackSymbol(self.symbol,self.embedding,self.predicted)
        s.has_to_move = self.has_to_move
        s.range       = self.range.copy()
        return s
    
    def schedule_movement(self,flag = True):
        """
        Internal method, flags the symbol for movement
        Args:
            flag (bool) : flags the symbol for movement (True) or stops movement (False) 
        """
        #TODO : check whether a full copy wouldn't be safer here
        self.has_to_move = flag

    def complete(self):
        """
        Internal method, setting the symbol as completed
        """
        #TODO : check whether a full copy wouldn't be safer here
        self.predicted = False
        
    def __str__(self):
        s =  '*%s'%(self.symbol,) if self.predicted else '%s*'%(self.symbol,)
        if self.has_to_move:
            s =  'm[%s]'%(s,)        
        return s

def print_config(config):
    S,B,n,stack_state,lab_state = config 
    return '(%s;%s;%d;%s)'%(','.join([ str(s) for s in S ]),str(B),n,lab_state)
    

class DiscoRNNGparser:
    """
    This is discontinuous RNNG with pre-order tree traversal and a move action
    """        
    #action codes
    SHIFT           = '<S>'
    OPEN            = '<O>'
    CLOSE           = '<C>'
    TERMINATE       = '<T>'
    MOVE            = '<M>' 

    #labelling states
    WORD_LABEL      = '@w'
    NT_LABEL        = '@n'
    NO_LABEL        = '@'
    MOVE_LABEL      = '@m'


    #special tokens
    UNKNOWN_TOKEN = '<UNK>'
    START_TOKEN   = '<START>'
    
    def __init__(self,stack_embedding_size=300,word_embedding_size=300,stack_hidden_size=300,brown_file='toto.brown'):

        self.brown_file = brown_file
        self.stack_embedding_size = stack_embedding_size
        self.word_embedding_size  = stack_embedding_size
        self.stack_hidden_size    = stack_hidden_size
 
    def allocate_conditional_params(self):
        """ 
        This allocates memory for the conditional model parameters
        """
        self.cond_model                     = dy.ParameterCollection()
        
        self.cond_nonterminals_embeddings   = self.cond_model.add_lookup_parameters((self.nonterminals.size(),self.stack_embedding_size)) 
        self.cond_word_embeddings           = self.cond_model.add_lookup_parameters((self.lexicon.size(),self.word_embedding_size)) 

        self.cond_structural_W             = self.cond_model.add_parameters((self.actions.size(),self.stack_hidden_size+self.stack_embedding_size))         
        self.cond_structural_b             = self.cond_model.add_parameters((self.actions.size()))
        
        #self.word_softmax                 = dy.ClassFactoredSoftmaxBuilder(self.stack_hidden_size,self.brown_file,self.lexicon.words2i,self.cond_model,bias=True)

        self.cond_nonterminals_W            = self.cond_model.add_parameters((self.nonterminals.size(),self.stack_hidden_size+self.stack_embedding_size))   
        self.cond_nonterminals_b            = self.cond_model.add_parameters((self.nonterminals.size()))

        self.cond_move                      = self.cond_model.add_parameters((1,self.stack_hidden_size+self.stack_embedding_size))
        
        #stack_lstm
        self.cond_rnn                      = dy.LSTMBuilder(2,self.stack_embedding_size, self.stack_hidden_size,self.cond_model)          
 
        self.cond_tree_fwd                 = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.cond_model)        
        self.cond_tree_bwd                 = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.cond_model)        
        self.cond_tree_W                   = self.cond_model.add_parameters((self.stack_embedding_size,self.stack_hidden_size*2))
        self.cond_tree_b                   = self.cond_model.add_parameters((self.stack_embedding_size))

        #specific to the cond model
        self.lexer_rnn_bwd                 = dy.LSTMBuilder(1,self.stack_embedding_size, self.word_embedding_size,self.cond_model)   
        
    def allocate_generative_params(self):
        """ 
        This allocates memory for the generative model parameters
        """
        self.gen_model                     = dy.ParameterCollection()
        
        self.gen_nonterminals_embeddings   = self.gen_model.add_lookup_parameters((self.nonterminals.size(),self.stack_embedding_size)) 
        self.gen_word_embeddings           = self.gen_model.add_lookup_parameters((self.lexicon.size(),self.word_embedding_size)) 

        self.gen_structural_W              = self.model.add_parameters((self.actions.size(),self.stack_hidden_size))         
        self.gen_structural_b              = self.model.add_parameters((self.actions.size()))
        
        self.word_softmax                  = dy.ClassFactoredSoftmaxBuilder(self.stack_hidden_size,self.brown_file,self.lexicon.words2i,self.cond_model,bias=True)

        self.gen_nonterminals_W            = self.gen_model.add_parameters((self.nonterminals.size(),self.stack_hidden_size))   
        self.gen_nonterminals_b            = self.gen_model.add_parameters((self.nonterminals.size()))

        self.gen_move                      = self.gen_model.add_parameters((1,self.stack_hidden_size+self.stack_embedding_size))
        
        #stack_lstm
        self.gen_rnn                       = dy.LSTMBuilder(2,self.stack_embedding_size, self.stack_hidden_size,self.gen_model)          
 
        self.gen_tree_fwd                  = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.gen_model)        
        self.gen_tree_bwd                  = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.gen_model)        
        self.gen_tree_W                    = self.gen_model.add_parameters((self.stack_embedding_size,self.stack_hidden_size*2))
        self.gen_tree_b                    = self.gen_model.add_parameters((self.stack_embedding_size))
    
    #TRANSITION SYSTEM AND ORACLE
    def init_configuration(self,N):
        """ 
        Inits a starting configuration. A configuration is 5-uple
        S: is the stack
        B: the buffer
        stack_mem: the current state of the stack lstm
        lab_state: the labelling state of the configuration
        
        Args:
           N   (int): the length of the input sequence
        """
        stack_state = self.cond_rnn.initial_state()
        w0          = self.cond_word_embeddings[self.lexicon.index(DiscoRNNGparser.START_TOKEN)]
        stack_state = stack_state.add_input(w0)
        return ([ ] ,tuple(range(N)),0, stack_state, DiscoRNNGparser.NO_LABEL)

    def shift_action(self,configuration):
        """
        This performs a shift action.
        That is the parser commits itself to generate a word at the next step.
        Args:
           configuration (tuple) : a configuration frow where to shift
        Returns: 
           tuple. a configuration resulting from shift 
        """
        S,B,n,stack_state,lab_state = configuration
        return (S,B,n,stack_state,DiscoRNNGparser.WORD_LABEL)
     
    def generate_word(self,configuration,sentence):
        """
        This generates a word (performs the actual shifting).
        Args:
           configuration (tuple) :  a configuration frow where to generate a word
           sentence       (list) :  a list of strings, the sentence tokens
        Returns:
           tuple. a configuration after word generation
        """
        S,B,n,stack_state,lab_state = configuration
        
        shifted     = sentence[B[0]]
        embedding   = self.cond_word_embeddings[self.lexicon.index(shifted)]
        stack_state = stack_state.add_input(embedding)
        #print('added shift embedding',shifted,embedding.npvalue())
        return (S + [StackSymbol(B[0],embedding,predicted=False,sym_range=[B[0]])],B[1:],n,stack_state,DiscoRNNGparser.NO_LABEL)
 
    def open_action(self,configuration):
        """
        Args:
           configuration (tuple): a configuration
        Returns:
           A configuration
        """
        S,B,n,stack_state,lab_state = configuration
        return (S,B,n,stack_state,DiscoRNNGparser.NT_LABEL)
    
    def open_nonterminal(self,configuration,label):
        """
        The nonterminal labelling action. This adds an open nonterminal on the stack under the stack top (left corner style inference)
        
        Arguments:
            configuration (tuple) : a configuration where to perform the labelling
            label         (string): the nonterminal label
        Returns:
            tuple. A configuration resulting from the labelling
        """
        S,B,n,stack_state,lab_state = configuration

        embedding   = self.cond_nonterminals_embeddings[self.nonterminals.index(label)]
        stack_state = stack_state.add_input(embedding)
        #print('added nt embedding',label,embedding.npvalue())
        return (S + [StackSymbol(label,embedding,predicted=True,sym_range=[B[0]])],B,n + 1,stack_state,DiscoRNNGparser.NO_LABEL) 
    
    def close_action(self,configuration): 
        """
        This actually executes the RNNG CLOSE action.
        Args:
           configuration (tuple): a configuration frow where to perform open
        Returns:
           tuple. A configuration resulting from closing the constituent
        """
        S,B,n,stack_state,lab_state = configuration
        newS = S[:]
        closed_symbols = []
        moved_symbols  = []
        complete_range = set() 
        
        while not (newS[-1].predicted and not newS[-1].has_to_move):

            stack_state = stack_state.prev()
            symbol = newS.pop() 

            if symbol.has_to_move:
                symbol.schedule_movement(False)
                moved_symbols.append(symbol)
            else:
                closed_symbols.append(symbol)
                if symbol.range:
                    complete_range = complete_range | set(symbol.range)

        stack_state = stack_state.prev()      
        completeNT = newS.pop()  

        #computes the tree embedding of the completed stuff
        fwd_state = self.cond_tree_fwd.initial_state()  
        fwd_state = fwd_state.add_input(self.cond_nonterminals_embeddings[self.nonterminals.index(completeNT.symbol)])
        for SYM in reversed(closed_symbols):
            fwd_state = fwd_state.add_input(SYM.embedding)
            
        bwd_state = self.cond_tree_bwd.initial_state()  
        bwd_state = bwd_state.add_input(self.cond_nonterminals_embeddings[self.nonterminals.index(completeNT.symbol)])
        for SYM in closed_symbols:
            bwd_state = bwd_state.add_input(SYM.embedding)

        tree_h         = dy.concatenate([self.ifdropout(fwd_state.output()),self.ifdropout(bwd_state.output())])
        tree_embedding = dy.rectify(self.cond_tree_W * tree_h + self.cond_tree_b)
        
        completeNT.complete()
        completeNT.range     = complete_range 
        completeNT.embedding = tree_embedding
        newS.append(completeNT)
        stack_state = stack_state.add_input(tree_embedding)
        
        #updates the stack state when putting back the moved elements
        newS.extend(reversed(moved_symbols))
        for SYM in reversed(moved_symbols):
             stack_state = stack_state.add_input(SYM.embedding)
             
        return (newS,B,n-1,stack_state,DiscoRNNGparser.NO_LABEL)

        
    def move_action(self,configuration,stack_idx):
        """
        This actually schedules a symbol down in the stack for movement
        Args:
           configuration (tuple) : a configuration
           stack_idx       (int) : the index in the stack of the element to move (top has index 0) 
        Returns:
           Tuple. A configuration resulting from moving the constituent
        """
        S,B,n,stack_state,lab_state = configuration
        S[-stack_idx-1].schedule_movement(True)
        return (S,B,n,stack_state,DiscoRNNGparser.MOVE_LABEL)

    
    def static_oracle(self,ref_root,global_root,sentence,configuration=None):
        """
        Generates a list of configurations and returns a list of actions to exec given a ref tree
        Args: 
          ref_root    (DiscoTree): the local root reference node.
          global_root (DiscoTree): the global root of the reference tree.
          sentence         (list): a list  of strings, the tokens
        KwArgs:
          configuration   (tuple): a configuration tuple or None at init
        Returns: 
          A list of actions, the last configuration.
        """       
        def occurs_predicted(ref_node,configuration):  # (occur check #1)
            #returns True if predicted node already on the stack
            S,B,n,stack_state,lab_state = configuration
            lc_idx = ref_node.left_corner() 
            for node in reversed(S): 
                if not node.predicted and min(node.range) == lc_idx: 
                    #print('**occ**',node.symbol,print_config(configuration))
                    #print('occ predicted',ref_node.label)
                    return True 
            return False
         
        def occurs_completed(ref_node,configuration):  # (occur check #2)
            #returns True if completed node already on the stack

            #print(print_config(configuration))
            S,B,n,stack_state,lab_state = configuration
            for elt in S:
                if not elt.predicted and ref_node.is_dominated_by(elt.range):
                    #print('occ completed',ref_node.label)
                    return True
            return False
    
            #max_incr_idx = max( [ max(elt.range) for elt in S if not elt.predicted] + [-1] )
            #ref_idx      = ref_node.right_corner()
            #if not ref_idx > max_incr_idx: #ERROR : move back in stack and search for an element with same label and same range !
            #    print('occ completed',ref_node.label)
            #    return True
            #return False
        
        if configuration is None:                       #init
            N = len(ref_root.words())
            configuration = self.init_configuration(N)   
            #print( print_config(configuration) )

        if ref_root.is_leaf(): 
            configuration = self.shift_action(configuration)
            configuration = self.generate_word(configuration,sentence)
            #print(print_config(configuration))
            S,B,n,stack_state,lab_state = configuration
            sh_word                     = sentence[S[-1].symbol]
            act_list                    = [DiscoRNNGparser.SHIFT,sh_word]
            return (act_list, configuration) 
        else:
            ##Recursive processing
            #A. Root 
            act_list = []
            if not occurs_predicted(ref_root,configuration):
                configuration = self.open_action(configuration) 
                configuration = self.open_nonterminal(configuration,ref_root.label)
                #print('OPENING',ref_root.label)
                #print(print_config(configuration))
                act_list.extend([DiscoRNNGparser.OPEN,ref_root.label])
            #else:
                #print('ALREADY OPEN',ref_root.label)
                
            #B. Recursive calls 
            for child in ref_root.covered_nodes(global_root):
                if not occurs_completed(child,configuration): #occur check : recursive call if and only if not already in
                    #non local extra processing
                    local = ref_root.dominates(child.range) 
                    if not local: 
                        #print('non loc call',child.label,'for root ',ref_root.label)
                        for ancestor in global_root.get_lc_ancestors(child.range):
                            #print('anc',ancestor.label)
                            if ancestor is not child:
                                configuration = self.open_action(configuration) 
                                configuration = self.open_nonterminal(configuration,ancestor.label)
                                #print(print_config(configuration))
                                act_list.extend([DiscoRNNGparser.OPEN,ancestor.label]) 

                    local_actions, configuration = self.static_oracle(child,global_root,sentence,configuration)
                    act_list.extend(local_actions)


            #C. Perform moves
            S,B,n,stack_state,lab_state = configuration
            for stack_idx, stack_elt in enumerate(reversed(S)):
                local = ref_root.dominates(stack_elt.range)
                #print(ref_root.range,stack_elt.range)
                if stack_elt.predicted and local:
                    break
                if not local:
                    #print('move',stack_elt.symbol)
                    #assert(stack_idx > 0) -> nope. there exists cases where this assertion does not hold (!)
                    configuration = self.move_action(configuration,stack_idx)
                    act_list.extend([DiscoRNNGparser.MOVE,stack_idx])       
                    #print(print_config(configuration))

            #D. Close
            #print("CLOSE",ref_root.label)  
            configuration = self.close_action(configuration)
            #print(print_config(configuration))
            act_list.append(DiscoRNNGparser.CLOSE)
            return (act_list,configuration)

        
    def deriv2tree(self,derivation):
        """
        Generates a discontinuous tree from the derivation
        Args:
           derivation (list): a list of actions as strings
        Returns:
          DiscoTree. The root of the tree
        """
        StackElt = namedtuple('StackElt',['symbol','predicted','has_to_move'])

        stack = []  
        inc_index   = 0
        prev_action = None
        for action in derivation:
            
            if prev_action   == DiscoRNNGparser.SHIFT:
                stack.append( StackElt(symbol=DiscoTree(action,child_index = inc_index), predicted=False,has_to_move=False) )
                inc_index += 1
                
            elif prev_action == DiscoRNNGparser.OPEN: 
                stack.append( StackElt(symbol=action, predicted=True,has_to_move=False) ) 
                
            elif prev_action == DiscoRNNGparser.MOVE: #sets the move on the stack

                sym,pred,mov = stack[ -int(action)-1 ]
                stack[ -int(action)-1 ] = StackElt(symbol=sym,predicted=pred,has_to_move=True)
                
            elif action == DiscoRNNGparser.CLOSE:
                nmoves   = 0
                children = [ ]
                moved    = [ ]
                while stack:
                    node = stack.pop()
                    if node.predicted and not node.has_to_move:
                        stack.append( StackElt(symbol=DiscoTree(node.symbol,children),predicted=False,has_to_move=False) )
                        stack.extend(reversed(moved)) 
                        break 
                    if node.has_to_move:
                        sym,pred,mov = node
                        moved.append( StackElt(symbol=sym,predicted=pred,has_to_move=False)  )
                    else:
                        children.append(node.symbol)
            prev_action = action
            
        root = stack.pop()
        return root.symbol

    def code_lexicon(self,treebank):
        """
        Builds indexes for word symbols found in the treebank
        """
        known_vocabulary = [ ]
        for tree in treebank:
            known_vocabulary.extend( tree.words() )
            
        known_vocabulary = get_known_vocabulary(known_vocabulary,vocab_threshold=-1)#change this
        known_vocabulary.add(DiscoRNNGparser.START_TOKEN)
        self.brown_file  = normalize_brown_file(self.brown_file,known_vocabulary,self.brown_file+'.unk',UNK_SYMBOL=DiscoRNNGparser.UNKNOWN_TOKEN)
        self.lexicon     = SymbolLexicon( list(known_vocabulary),unk_word=DiscoRNNGparser.UNKNOWN_TOKEN)
        return self.lexicon
     
    def code_nonterminals(self,train_treebank,dev_treebank):
        """
        Extracts the nonterminals from a treebank and codes them on integers as a lexicon object.
        
        Args:
           train_treebank   (list) : a list of trees  where to extract the non terminals from
           dev_treebank   (list) : a list of trees  where to extract the non terminals from
        Returns:
           SymbolLexicon. The bijective encoding
        """
        nonterminals = set([])
        for tree in train_treebank:
            nonterminals.update(tree.collect_nonterminals())
        for tree in dev_treebank:
            nonterminals.update(tree.collect_nonterminals())
        self.nonterminals = SymbolLexicon(list(nonterminals))
        return self.nonterminals

    def code_struct_actions(self):
        """
        Codes the structural actions on integers and generates bool masks
        Returns:
            SymbolLexicon. The bijective encoding
        """
        #Structural actions are coded on the first slots
        #The last slots are implicitly allocated to move actions
        self.actions         = SymbolLexicon([DiscoRNNGparser.SHIFT,DiscoRNNGparser.OPEN,DiscoRNNGparser.CLOSE,DiscoRNNGparser.TERMINATE])
        return self.actions

    def allowed_structural_actions(self,configuration): 
        """
        Returns the list of structural actions allowed given this configuration.
        Arguments:
           configuration          (tuple): a configuration
        Returns:
           a list. Indexes of the allowed actions
        """
        S,B,n,stack_state,lab_state = configuration
        
        #upper bound on num moves
        nmoves = 0
        for idx,stack_elt in enumerate(reversed(S)):
            if stack_elt.predicted and not stack_elt.has_to_move:
                nmoves = idx+1 
                break
        MASK = np.array([True]*self.actions.size()+[False]*nmoves)  if n < 1 else  np.array([True]*self.actions.size()+[True]*nmoves) 
        if not B:
            #last condition prevents unaries
            MASK[self.actions.index(DiscoRNNGparser.OPEN)] = False  
        if B or n != 0 or len(S) > 1:
            MASK[self.actions.index(DiscoRNNGparser.TERMINATE)] = False  
        if not B or (S and n == 0):
            MASK[self.actions.index(DiscoRNNGparser.SHIFT)] = False  
        if not S or n < 1 or (len(S) >=1 and S[-1].predicted and S[-1].symbol in self.nonterminals):
            # Last condition prevents unaries
            # Exceptional unaries are allowed on top of terminals symbols only
            MASK[self.actions.index(DiscoRNNGparser.CLOSE)] = False  

        allowed_idxes = [idx for idx, mask_val in enumerate(MASK) if mask_val]
        return allowed_idxes

    def ifdropout(self,expression):
        """
        Applies dropout to a dynet expression only if dropout > 0.0.
        """
        return dy.dropout(expression,self.dropout) if self.dropout > 0.0 else expression

    def dynamic_move_matrix(self,stack,stack_state,buffer_embedding,conditional):
        """
        Dynamically computes the score of each possible valid move action
        Args:
            stack                       (list): a list of StackELements (the stack of a configuration)
            stack_state          (dynet stuff): pointer to an rnn state
            buffer_embedding(dynet expression): the embedding of the first word in the buffer
            conditional                (bool) : bool stating if we use conditional or generative params
        Returns: 
            A dynet expression
        """
        local_state  = stack_state
        stack_scores = [ ] 
        
        for idx,stack_elt in enumerate(reversed(stack)):
            H =  dy.concatenate([local_state.output(),buffer_embedding])
            if conditional:
                stack_scores.append( self.cond_move * H )
            else:
                stack_scores.append( self.gen_move * H )
                
            if stack_elt.predicted and not stack_elt.has_to_move : #check this condition:up to where can we move ?
                break
        return dy.concatenate(stack_scores) if stack_scores else stack_scores


    
    def predict_action_distrib(self,configuration,sentence,word_encodings,conditional):
        """
        Predicts the log distribution for next actions from the current configuration.
        Args:
          configuration           (tuple): the current configuration
          sentence                 (list): a list of string, the tokens
          word_encodings           (list): a list of embeddings for the tokens. None in case of generative inference
          conditional              (bool): flag stating whether to perform conditional or generative inference
        Returns:
            a list of couples (action, log probability). The list is empty if the parser is trapped (aka no action is possible).
            currently returns a zip generator.
        """
        def code2action(act_idx): 
            return [self.MOVE,act_idx-self.actions.size()]  if type(act_idx) == int else  self.actions.wordform(action_idx)
     
        S,B,n,stack_state,lab_state = configuration

        if lab_state == DiscoRNNGparser.WORD_LABEL:
            next_word     = (sentence[B[0]])
            if conditional:
                return [(next_word,0)] # in the discriminative case words are given and have prob = 1.0
            else:
                pass 
            # TODO in the generative case...
            #next_word_idx = self.lexicon.index(next_word)
            #return [(next_word,-self.word_softmax.neg_log_softmax(dy.rectify(stack_state.output()),next_word_idx).value())]
        elif lab_state == DiscoRNNGparser.NT_LABEL :
            if conditional:
                word_idx = B[0] if B else -1
                H =  dy.concatenate([stack_state.output(),word_encodings[word_idx]])
                logprobs = dy.log_softmax(self.cond_nonterminals_W  * dy.rectify(H)  + self.cond_nonterminals_b).value()
                return zip(self.nonterminals.i2words,logprobs)
            else:
                pass
        elif lab_state == DiscoRNNGparser.NO_LABEL :
            if conditional:
                ref_idx          = self.actions.size() + ref_action if type(ref_action) == int else self.actions.index(ref_action)
                restr_mask       = self.allowed_structural_actions(configuration)

                if restr_mask:
                    word_idx         = B[0] if B else -1
                    buffer_embedding = word_encodings[word_idx] 

                    hidden_input     = dy.concatenate([stack_state.output(),word_encodings[word_idx]])
                    static_scores    = self.cond_structural_W  * self.ifdropout(dy.rectify(hidden_input))  + self.cond_structural_b
                    move_scores      = self.dynamic_move_matrix(S,stack_state,buffer_embedding,conditional)
                    all_scores        = dy.concatenate([static_scores,move_scores]) if move_scores else static_scores
                    logprobs         = dy.log_softmax(all_scores,restr_mask).value()
                    return [ (code2action(action_idx),logprob) for action_idx,logprob in zip(range(self.actions.size()),logprobs) if action_idx in restr]
            else:
                pass #add generative stuff here
                
        return [ ]
 
     
    def eval_action_distrib(self,configuration,sentence,word_encodings,ref_action,conditional): 
        """
        Evaluates the model predictions against the reference data.
        Args:
          configuration   (tuple): the current configuration
          sentence         (list): a list of string, the tokens
          word_encodings   (list): a list of embeddings for the tokens. None in case of generative inference
          ref_action     (string): the reference action.
          conditional              (bool): flag stating whether to perform conditional or generative inference
        Returns: 
            a dynet expression. The loss (NLL) for this action
        """
        S,B,n,stack_state,lab_state = configuration
        #print(stack_state.output().npvalue())
        if lab_state == DiscoRNNGparser.WORD_LABEL:
            if conditional:
                #in the discriminative case the word is given and has nll = 0
                nll = dy.scalarInput(0.0)  
            else:
                pass
            #TODO : in the generative case
            #ref_idx  = self.lexicon.index(ref_action)
            #nll =  self.word_softmax.neg_log_softmax(self.ifdropout(dy.rectify(stack_state.output())),ref_idx) 

        elif lab_state == DiscoRNNGparser.NT_LABEL:
            
            if conditional:
                ref_idx  = self.nonterminals.index(ref_action)
                word_idx = B[0] if B else -1
                H        =  dy.concatenate([stack_state.output(),word_encodings[word_idx]])
                nll      = dy.pickneglogsoftmax(self.cond_nonterminals_W  * self.ifdropout(dy.rectify(H)) + self.cond_nonterminals_b,ref_idx)
            else:
                pass
            
        elif lab_state == DiscoRNNGparser.NO_LABEL:
            #TODO (for predict_action_disrib)
            if conditional:
                ref_idx          = self.actions.size() + ref_action if type(ref_action) == int else self.actions.index(ref_action)
                restr_mask       = self.allowed_structural_actions(configuration)

                word_idx         = B[0] if B else -1
                buffer_embedding = word_encodings[word_idx] 

                hidden_input     = dy.concatenate([stack_state.output(),word_encodings[word_idx]])
                static_scores    = self.cond_structural_W  * self.ifdropout(dy.rectify(hidden_input))  + self.cond_structural_b
                move_scores      = self.dynamic_move_matrix(S,stack_state,buffer_embedding,conditional)
                all_scores        = dy.concatenate([static_scores,move_scores]) if move_scores else static_scores
                nll              = -dy.pick(dy.log_softmax(all_scores,restr_mask),ref_idx)
            else:
                pass
        else:
            print('error in evaluation')
        return nll


    def eval_derivation(self,ref_derivation,sentence,word_encodings,backprop=True):
        """
        Evaluates the model predictions against the reference derivation

        Args:
          ref_derivation                (list) : a reference derivation
          sentence                      (list) : a list of strings (words)
          word_encodings                (list) : a list of dynet expressions (word embeddings)
        Kwargs:
          backprop                       (bool): a flag telling if we perform backprop or not
        Returns:
          RuntimeStats. the model NLL, the word only NLL, the size of the derivations, the number of predicted words 
        """
        dropout = self.dropout
        if not backprop:
            self.dropout = 0.0
        
        runstats = RuntimeStats('NLL','lexNLL','N','lexN')
        runstats.push_row() 
        
        runstats['lexN'] = len(sentence)
        runstats['N']    = len(ref_derivation)

        all_NLL     = [] #collects the local losses in the batch
        lexical_NLL = [] #collects the local losses in the batch (for word prediction only)
        
        configuration = self.init_configuration( len(sentence) )
        prev_action = None
        for ref_action in ref_derivation:
              
            S,B,n,stack_state,lab_state = configuration                

            nll =  self.eval_action_distrib(configuration,sentence,word_encodings,ref_action,True)
            all_NLL.append( nll )
            
            if lab_state == DiscoRNNGparser.WORD_LABEL:
                configuration = self.generate_word(configuration,sentence)
                lexical_NLL.append(nll)
            elif lab_state == DiscoRNNGparser.NT_LABEL:
                configuration = self.open_nonterminal(configuration,ref_action)
            elif prev_action == DiscoRNNGparser.MOVE:
                configuration = self.move(configuration,int(ref_action))
            elif ref_action == DiscoRNNGparser.CLOSE:
                configuration = self.close_action(configuration)
            elif ref_action == DiscoRNNGparser.OPEN:
                configuration = self.open_action(configuration)
            elif ref_action == DiscoRNNGparser.SHIFT:
                configuration = self.shift_action(configuration)
            elif ref_action == DiscoRNNGparser.TERMINATE:
                pass
            prev_action = ref_action
              
        loss     = dy.esum(all_NLL)
        lex_loss = dy.esum(lexical_NLL)
        
        runstats['NLL']    = loss.value() 
        runstats['lexNLL'] = lex_loss.value() 
         
        if backprop:
            loss.backward()
            try:
                self.trainer.update()
            except RuntimeError:
                print('\nGradient exploded, batch update aborted...')
        else:
            self.dropout = dropout
            
        return runstats
    
    def eval_sentence(self,ref_tree,conditional,backprop=True):
        #add an option for training the generative here
        """
        Evaluates the model predictions against the reference data.
        and optionally performs backpropagation. 
        The function either takes a single tree or a batch of trees (as list) for evaluation.
        Args:
          ref_tree                  (ConsTree) : a reference tree or a single tree.
        Kwargs:
          conditional                    (bool): a flag telling if we use the conditional or the generative model
          backprop                       (bool): a flag telling if we perform backprop
        Returns:
          RuntimeStats. the model NLL, the word only NLL, the size of the derivations, the number of predicted words on this batch
        """
        dy.renew_cg()
        
        sentence = ref_tree.words()

        if conditional:
            #encode lookahead 
            lex_state       = self.lexer_rnn_bwd.initial_state()
            start_embedding = self.cond_word_embeddings[ self.lexicon.index(DiscoRNNGparser.START_TOKEN) ]
            lex_state       = lex_state.add_input(start_embedding)
            word_embeddings = [self.cond_word_embeddings[self.lexicon.index(w)] for w in reversed(sentence) ]
            word_encodings  = lex_state.transduce(word_embeddings)
            word_encodings.reverse()  
             
            derivation,last_config = self.static_oracle(ref_tree,ref_tree,sentence)
            return self.eval_derivation(derivation,sentence,word_encodings,backprop)
        else:
            pass
            #add stuff for training the generative model here
        
    def train_model(self,train_stream,dev_stream,modelname,lr=0.1,epochs=20,batch_size=1,dropout=0.3):
        """
        Estimates the parameters of a model from a treebank.
        Args:
           train_stream  (stream): a stream open on a treebank file 
           dev_stream    (stream): a stream open on a treebank file 
        """
        #preprocessing
        train_treebank = [ ]
        idx = 0
        for line in train_stream:
            t = DiscoTree.read_tree(line)
            t.strip_tags()
            t.close_unaries()
            train_treebank.append(t)
            print(t)
            break
            idx += 1  #just 10 trees for now 
            if idx > 10:
                break
            
        #dev_treebank = []
        #for line in dev_stream:
        #    t = DiscoTree.read_tree(line)
        #    t.strip_tags()
        #    t.close_unaries()
        #    dev_treebank.append(t)
        dev_treebank = train_treebank   #just 1 tree for now 
            
        self.code_lexicon(train_treebank)
        self.code_nonterminals(train_treebank,dev_treebank)
        self.code_struct_actions()
        self.allocate_conditional_params()

        #Training
        self.dropout = dropout
        self.trainer = dy.SimpleSGDTrainer(self.cond_model,learning_rate=lr)
        min_nll      = np.inf

        ntrain_sentences = len(train_treebank)
        ndev_sentences   = len(dev_treebank)

        train_stats = RuntimeStats('NLL','lexNLL','N','lexN')
        valid_stats = RuntimeStats('NLL','lexNLL','N','lexN')
        
        for e in range(epochs):

            train_stats.push_row()
            for idx,tree in enumerate(train_treebank):
                train_stats += self.eval_sentence(tree,conditional=True,backprop=True)
                sys.stdout.write('\r===> processed %d training trees'%(idx+1))

            NLL,lex_NLL,N,lexN = train_stats.peek()            
            print('\n[Training]   Epoch %d, NLL = %f, lex-NLL = %f, PPL = %f, lex-PPL = %f'%(e,NLL,lex_NLL,np.exp(NLL/N),np.exp(lex_NLL/lexN)),flush=True)

            valid_stats.push_row()
            for idx,tree in enumerate(dev_treebank):
                valid_stats += self.eval_sentence(tree,conditional=True,backprop=False)
 
            NLL,lex_NLL,N,lexN = valid_stats.peek()    
            print('[Validation] Epoch %d, NLL = %f, lex-NLL = %f, PPL = %f, lex-PPL = %f'%(e,NLL,lex_NLL, np.exp(NLL/N),np.exp(lex_NLL/lexN)),flush=True)
            print()
            if NLL < min_nll:
                pass
                #self.save_model(modelname)
        
if __name__ == '__main__':

    p = DiscoRNNGparser(brown_file='kk.brown')

    tstream = open('negra/test.mrg')
    dstream = open('negra/dev.mrg')
    p.train_model(tstream,tstream,'test',lr=0.1,epochs=100,dropout=0.0)
    tstream.close()
    dstream.close()

    exit(0)
    
    t = DiscoTree.read_tree('(S (NP 0=John) (VP (VB 1=eats) (NP (DT 2=an) (NN 3=apple))) (PONCT 4=.))')
    print(t)
    wordlist = t.words()
    print(wordlist)
    print()

    p = DiscoRNNGparser()
    D,C = p.static_oracle(t,t,wordlist)
    print(D)
    print(p.deriv2tree(D))
    print()
    
    t = DiscoTree.read_tree('(S (VP (VB 0=is) (JJ 2=rich)) (NP 1=John) (PONCT 3=?))')
    print(t)
    wordlist = t.words()
    print(wordlist)
    print()

    p = DiscoRNNGparser()

    D,C = p.static_oracle(t,t,wordlist)
    print(D)
    print(p.deriv2tree(D))
    print() 

    
    t2 = DiscoTree.read_tree("(ROOT (SBARQ (SQ (VP (WHADVP (WRB 0=Why)) (VB 4=cross) (NP (DT 5=the) (NN 6=road))) (VBD 1=did) (NP (DT 2=the) (NN 3=chicken))) (PONCT 7=?)))")
    print(t2,'gap_degree',t2.gap_degree())
    wordlist = t2.words()
    print(wordlist)
    print()
 
    D,C = p.static_oracle(t2,t2,wordlist)
    print(D)
    print(p.deriv2tree(D))
    print()
    
    t3 =  DiscoTree.read_tree('(S (X (A 0=a)  (A 3=a))  (Y (B 1=b) (B 4=b)) (Z (C 2=c) (C 5=c)))')
    print(t3,'gap_degree',t3.gap_degree())
    wordlist = t3.words()
    print(wordlist)
    print()

    D,C = p.static_oracle(t3,t3,wordlist)
    print(D)
    print(p.deriv2tree(D))
    print() 

    #t4 = DiscoTree.read_tree('(ROOT (CS (S (CNP (NP (ART 0=Die) (MPN (NE 1=Rolling) (NE 2=Stones))) (KON 3=oder) (MPN (NE 4=Led) (NE 5=Zeppelin))) (VAFIN 6=haben) (NP (ADV 7=auch) (PIAT 8=keinen) (NE 9=Grammy))) ($, 10=,) (KON 11=und) (S (NP (PDS 12=die) (AP (PIAT 16=mehr) (NP (KOKOM 18=als) (PPER 19=ich)))) (VAFIN 13=hätten) (VP (PPER 14=ihn) (ADV 15=sicherlich) (VVPP 17=verdient)))) (D. 20=.) (D[ 21="))')
    t4 = DiscoTree.read_tree('(S (NP (PDS 0=die) (AP (PIAT 4=mehr) (NP (KOKOM 6=als) (PPER 7=ich)))) (VAFIN 1=hätten) (VP (PPER 2=ihn) (ADV 3=sicherlich) (VVPP 5=verdient)))')
    t4.close_unaries()
    wordlist = t4.words()
    print(wordlist)

    D,C = p.static_oracle(t4,t4,wordlist)
    print(D)
    print(p.deriv2tree(D))
    print()


    
