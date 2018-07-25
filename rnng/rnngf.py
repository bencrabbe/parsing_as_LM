"""
This class implements an RNNG parser with class factored word emissions.
"""
import sys
import dynet as dy
import numpy as np
import numpy.random as npr
import getopt
import warnings
import json

from collections   import Counter
from random        import shuffle
from constree      import *
from lexicons      import *
from proc_monitors import *
from rnng_params   import *

class StackSymbol:
    """
    A convenience class for stack symbols
    """
    PREDICTED = 1
    COMPLETED = 0
    
    def __init__(self,symbol,status,embedding):
        """
        Args:
             symbol           (string): a non terminal or a word
             status             (enum): predicted or completed
             embedding (dy.expression): a dynet expression being the embedding of the subtree dominated by this symbol (or word)
        """
        self.symbol,self.status,self.embedding = symbol,status,embedding

    def copy(self):
        return StackSymbol(self.symbol,self.status,self.embedding)

    def complete(self):
        c = self.copy()
        c.status = StackSymbol.COMPLETED
        return c
    
    def __str__(self):
        s =  '*%s'%(self.symbol,) if self.status == StackSymbol.PREDICTED else '%s*'%(self.symbol,)
        return s

class RNNGparser:
    """
    This is an RNNG parser with in-order tree traversal
    """
    #action codes
    SHIFT           = '<S>'
    OPEN            = '<O>'
    CLOSE           = '<C>'
    TERMINATE       = '<T>'
    
    #labelling states
    WORD_LABEL      = '@w'
    NT_LABEL        = '@n'
    NO_LABEL        = '@-'
    
    #special tokens
    UNKNOWN_TOKEN = '<UNK>'
    START_TOKEN   = '<START>'

    def __init__(self,brown_clusters,
                      max_vocabulary_size=10000,\
                      stack_embedding_size=100,
                      stack_memory_size=100,
                      word_embedding_size=100):
        """
        Args:
           brown_clusters       (str)  : a filename where to find brown clusters     
        Kwargs:
           max_vocabulary_size  (int)  : max number of words in the lexical vocab
           stack_embedding_size (int)  : size of stack lstm input 
           stack_memory_size    (int)  : size of the stack and tree lstm hidden layers
           word_embedding_size  (int)  : size of word embeddings
        """
        
        self.brown_file           = brown_clusters
        self.max_vocabulary_size  = max_vocabulary_size
        self.stack_embedding_size = stack_embedding_size
        self.stack_hidden_size    = stack_memory_size
        self.word_embedding_size  = word_embedding_size
        self.dropout              = 0.0

    def code_lexicon(self,treebank):
        """
        Codes a lexicon on integers indexes and generates a lexicon object.
        
        Args:
             treebank       (list) : a list of trees where to extract the words from
             max_vocab_size (int)  : the upper bound on the size of the 'real' vocabulary
        Returns:
             SymbolLexicon. The bijective encoding
        """
        lexicon = Counter()
        for tree in treebank:
            lexicon.update(tree.tokens())
        known_vocabulary = set([word for word, counts in lexicon.most_common(self.max_vocabulary_size)])
        known_vocabulary.add(RNNGparser.START_TOKEN)
        self.brown_file  = normalize_brown_file(self.brown_file,known_vocabulary,self.brown_file+'.unk',UNK_SYMBOL=RNNGparser.UNKNOWN_TOKEN)
        self.lexicon     = SymbolLexicon( list(known_vocabulary),unk_word=RNNGparser.UNKNOWN_TOKEN)
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
        self.actions         = SymbolLexicon([RNNGparser.SHIFT,RNNGparser.OPEN,RNNGparser.CLOSE,RNNGparser.TERMINATE])

        #Allocates masks
        self.open_mask       = np.array([True]*4)
        self.shift_mask      = np.array([True]*4)
        self.close_mask      = np.array([True]*4)
        self.terminate_mask  = np.array([True]*4)

        self.open_mask[self.actions.index(RNNGparser.OPEN)]           = False
        self.shift_mask[self.actions.index(RNNGparser.SHIFT)]         = False
        self.close_mask[self.actions.index(RNNGparser.CLOSE)]         = False
        self.terminate_mask[self.actions.index(RNNGparser.TERMINATE)] = False

        return self.actions

    @staticmethod
    def load_model(self,model_name):
        """
        Loads an RNNG parser from params at prefix model_name

        Args:
            model_name   (string): the prefix path for param files

        Returns:
            RNNGparser. An instance of RNNG ready to use.
        """
        hyperparams = json.loads(open(model_name+'.json').read())
        parser = RNNGparser(brown_clusters,
                            max_vocabulary_size=hyperparams['brown_file'],\
                            stack_embedding_size=hyperparams['stack_embedding_size'],\
                            stack_memory_size=hyperparams['stack_hidden_size'],\
                            word_embedding_size=hyperparams['word_embedding_size'])

        parser.lexicon      = SymbolLexicon.load(model_name+'.lex')
        parser.nonterminals = SymbolLexicon.load(model_name+'.nt')
        parser.code_struct_actions()
        parser.allocate_structure()
        parser.model.populate(model_name+".weights")

        return parser

    
    def save_model(self,model_name):
        """
        Saves the model params using the prefix model_name.

        Args:
            model_name   (string): the prefix path for param files
        """

        prefix_name = '/'.joinr['dirname','dirname']
        
        hyperparams = { 'brown_file':brown_file,\
                        'max_vocabulary_size':max_vocabulary_size,\
                        'stack_embedding_size':stack_embedding_size,\
                        'stack_hidden_size':stack_hidden_size,\
                        'word_embedding_size':word_embedding_size}
  
        jfile = open(model_name+'.json','w')
        jfile.dumps(hyperparams)
        jfile.close()

        self.model.save(model_name+'.weights')
        self.lexicon.save(model_name+'.lex')
        self.nonterminals.save(model_name+'.lex')
        
    
    #TRANSITION SYSTEM AND ORACLE
    
    def init_configuration(self,N):
        """
        A configuration is a 5-tuple (S,B,n,stack_mem,labelling?) where:
        
          S: is the stack
          B: the buffer
          n: the number of predicted constituents in the stack
          stack_mem: the current state of the stack lstm
          lab_state: the labelling state of the configuration
        
        This method creates an initial configuration, with empty stack, full buffer (as a list of integers and null score)
        Stacks are filled with non terminals of type StackSymbol.
        Buffers are only filled with terminals of type integer.

        Arguments:
           N  (int) : the length of the input sentence
        Returns:
           tuple. an initial configuration
        """
        stack_state = self.rnn.initial_state()
        e = self.word_embeddings[self.lexicon.index(RNNGparser.START_TOKEN)]
        stack_state = stack_state.add_input(e)
        return ([],tuple(range(N)),0,stack_state,RNNGparser.NO_LABEL)
    
    def shift_action(self,configuration):
        """
        This performs a shift action.
        That is the parser commits itself to generate a word at the next step.

        Arguments:
           configuration (tuple) : a configuration frow where to shift
        Returns:
           tuple. a configuration resulting from shift 
        """
        S,B,n,stack_state,lab_state = configuration
        return (S,B,n,stack_state,RNNGparser.WORD_LABEL)

    def generate_word(self,configuration,sentence):
        """
        This generates a word (performs the actual shifting).
        
        Arguments:
            configuration (tuple) :  a configuration frow where to generate a word
            sentence      (list)  :  a list of strings, the sentence tokens
        Returns:
           tuple. a configuration after word generation
        """
        S,B,n,stack_state,lab_state = configuration
        e = self.word_embeddings[self.lexicon.index(sentence[B[0]])]
        return (S + [StackSymbol(B[0],StackSymbol.COMPLETED,e)],B[1:],n,stack_state.add_input(e),RNNGparser.NO_LABEL)

    def open_action(self,configuration):
        """
        The Open action commits the parser to open a constituent without doing it immediately
        Arguments:
           configuration (tuple): a configuration frow where to perform open
        Returns:
           tuple. A configuration resulting from opening the constituent
        """
        S,B,n,stack_state,lab_state = configuration
        return (S,B,n,stack_state,RNNGparser.NT_LABEL)

    def label_nonterminal(self,configuration,Xlabel):
        """
        The nonterminal labelling action. This adds an open nonterminal on the stack under the stack top (left corner style inference)
        
        Arguments:
            configuration (tuple) : a configuration where to perform the labelling
            Xlabel        (string): the nonterminal label
        Returns:
            tuple. A configuration resulting from the labelling
        """
        S,B,n,stack_state,lab_state = configuration
        
        stack_top = S[-1]
        e = self.nonterminals_embeddings[self.nonterminals.index(Xlabel)]
        return (S[:-1] + [StackSymbol(Xlabel,StackSymbol.PREDICTED,e),stack_top],B,n+1,stack_state.add_input(e),RNNGparser.NO_LABEL)

    def close_action(self,configuration):
        """
        This is the RNNG CLOSE action.
        Arguments:
            configuration (tuple) : a configuration where to perform the closure
        Returns:
            tuple. A configuration resulting from the closure.
        """
        S,B,n,stack_state,lab_state = configuration

        assert(n > 0)
        
        newS = S[:]
        closed_symbols = []
        while newS[-1].status != StackSymbol.PREDICTED:
            closed_symbols.append(newS.pop())
            stack_state = stack_state.prev()
        newS[-1] = newS[-1].complete()
        stack_state = stack_state.prev()         #pops the NT embedding too
       
        #tree rnn
        fwd_state = self.tree_fwd.initial_state()  
        fwd_state = fwd_state.add_input(self.nonterminals_embeddings[self.nonterminals.index(newS[-1].symbol)])
        for SYM in reversed(closed_symbols):
            fwd_state = fwd_state.add_input(SYM.embedding)
            
        bwd_state = self.tree_bwd.initial_state()  
        bwd_state = bwd_state.add_input(self.nonterminals_embeddings[self.nonterminals.index(newS[-1].symbol)])
        for SYM in closed_symbols:
            bwd_state = bwd_state.add_input(SYM.embedding)

        tree_h         = dy.concatenate([fwd_state.output(),bwd_state.output()])
        tree_embedding = dy.rectify(self.tree_W * tree_h + self.tree_b)

        return (newS,B,n-1,stack_state.add_input(tree_embedding),RNNGparser.NO_LABEL)

    
    def static_inorder_oracle(self,ref_tree,sentence,configuration=None):
        """
        This generates a simple oracle derivation by performing an inorder traversal of the reference tree.
        The function simulates parsing and thus checks for oracle soundness and also generates reference configurations.

        Arguments:
            ref_tree (ConsTree)   : a local tree root
            sentence (list)       : a list of strings, the tokens of the sentence.
        Kwargs:
            configuration (tuple) : the current configuration
        Returns:
            (a list of actions,the resulting configuration). Actions in the derivation are coded as strings 
        """
        is_root = False
        if configuration is None:
            configuration = self.init_configuration(len(sentence))
            is_root = True
            
        if ref_tree.is_leaf():
            if not self.actions.index(RNNGparser.SHIFT) in self.allowed_structural_actions(configuration):
                print("oracle unsound <shift> ",configuration,ref_tree)
            configuration = self.shift_action(configuration)
            configuration = self.generate_word(configuration,sentence)
            return ( [RNNGparser.SHIFT, ref_tree.label], configuration)
        
        else:
            first_child = ref_tree.children[0]
            derivation, configuration = self.static_inorder_oracle(first_child,sentence,configuration)

            if not self.actions.index(RNNGparser.OPEN) in self.allowed_structural_actions(configuration):
                print('oracle unsound <open>',ref_tree)
            configuration = self.open_action(configuration)
            configuration = self.label_nonterminal(configuration,ref_tree.label)
            derivation.extend([RNNGparser.OPEN,ref_tree.label])
            
            for child in ref_tree.children[1:]:
                subderivation,configuration = self.static_inorder_oracle(child,sentence,configuration) 
                derivation.extend(subderivation)
                
            if not self.actions.index(RNNGparser.CLOSE) in self.allowed_structural_actions(configuration):
                print('oracle unsound <close>',ref_tree)
            configuration = self.close_action(configuration)
            derivation.append(RNNGparser.CLOSE)
            
        if is_root:
             derivation.append(RNNGparser.TERMINATE)
             
        return (derivation,configuration)
     
    def allowed_structural_actions(self,configuration):
        """
        Returns the list of structural actions allowed given this configuration.
        Arguments:
           configuration (tuple) : a configuration
        Returns:
           a list. Indexes of the allowed actions
        """        
        S,B,n,stack_state,lab_state = configuration 
        MASK = np.array([True] * self.actions.size())
        
        if not S or S[-1].status != StackSymbol.COMPLETED: 
            MASK *= self.open_mask
        if B or n > 0 or len(S) > 1:
            MASK *= self.terminate_mask
        if not B or (S and n == 0):
            MASK *= self.shift_mask
        if not S or n < 1: 
            MASK *= self.close_mask

        allowed_idxes = [idx for idx, mask_val in enumerate(MASK) if mask_val]
        return allowed_idxes

    def allocate_structure(self):
        """
        Allocates memory for the model parameters.

        Args:
           brown_clusters       (str)  : a filename where to find brown clusters     
        Kwargs:
           max_vocabulary_size  (int)  : max number of words in the lexical vocab
           stack_embedding_size (int)  : size of stack lstm input 
           stack_memory_size    (int)  : size of the stack and tree lstm hidden layers
           word_embedding_size  (int)  : size of word embeddings
           char_embedding_size  (int)  : size of char lstm input 
           char_memory_size     (int)  : size of the char lstm hidden layer
        """
        self.model                     = dy.ParameterCollection()

        #input
        self.nonterminals_embeddings   = self.model.add_lookup_parameters((self.nonterminals.size(),self.stack_embedding_size)) 
        self.word_embeddings           = self.model.add_lookup_parameters((self.lexicon.size(),self.word_embedding_size)) 

        #output
        self.structural_W             = self.model.add_parameters((self.actions.size(),self.stack_hidden_size))         
        self.structural_b             = self.model.add_parameters((self.actions.size()))

        self.word_softmax             = dy.ClassFactoredSoftmaxBuilder(self.stack_hidden_size,self.brown_file,self.lexicon.words2i,self.model,bias=True)

        self.nonterminals_W           = self.model.add_parameters((self.nonterminals.size(),self.stack_hidden_size))   
        self.nonterminals_b           = self.model.add_parameters((self.nonterminals.size()))

        #stack_lstm
        self.rnn                      = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)          
 
        #tree bi-lstm
        self.tree_fwd                 = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)        
        self.tree_bwd                 = dy.LSTMBuilder(1,self.stack_embedding_size, self.stack_hidden_size,self.model)        
        self.tree_W                   = self.model.add_parameters((self.stack_embedding_size,self.stack_hidden_size*2))
        self.tree_b                   = self.model.add_parameters((self.stack_embedding_size))

        
    def predict_action_distrib(self,configuration,sentence):
        """
        Predicts the log distribution for next actions from the current configuration.

        Args:
          configuration   (tuple): the current configuration
          sentence         (list): a list of string, the tokens

        Returns:
            a list of couples (action, log probability) or None if the parser is trapped.
        """
        S,B,n,stack_state,lab_state = configuration

        if lab_state == RNNGparser.WORD_LABEL:
            next_word     = (sentence[B[0]])
            next_word_idx = self.lexicon.index(next_word)
            return [(next_word,-self.word_softmax.neg_log_softmax(dy.rectify(stack_state.output()),next_word_idx).value())]
        elif lab_state == RNNGparser.NT_LABEL :
            logprobs = dy.log_softmax(self.nonterminals_W  * dy.rectify(stack_state.output())  + self.nonterminals_b).value()
            return list(zip(self.nonterminals.i2words,logprobs))
        elif lab_state == RNNGparser.NO_LABEL :
            restr = self.allowed_structural_actions(configuration)
            if restr:
                logprobs =  dy.log_softmax(self.structural_W  * dy.rectify(stack_state.output())  + self.structural_b,restr).value()
                return list(zip(self.actions.i2words,logprobs))
        print('prediction error, parser trapped')
        return None


    def eval_action_distrib(self,configuration,sentence,ref_action):
        """
        Evaluates the model predictions against the reference data.

        Args:
          configuration   (tuple): the current configuration
          sentence         (list): a list of string, the tokens
          ref_action     (string): the reference action.
          
        Returns:
            a dynet expression. The loss (NLL) for this action
        """
        S,B,n,stack_state,lab_state = configuration

        if lab_state == RNNGparser.WORD_LABEL :
            ref_idx  = self.lexicon.index(ref_action)
            nll =  self.word_softmax.neg_log_softmax(dy.rectify(stack_state.output()),ref_idx)
        elif lab_state == RNNGparser.NT_LABEL :
            ref_idx  = self.nonterminals.index(ref_action)
            nll = dy.pickneglogsoftmax(self.nonterminals_W  * dy.rectify(stack_state.output())  + self.nonterminals_b,ref_idx)
        elif lab_state == RNNGparser.NO_LABEL :
            ref_idx = self.actions.index(ref_action)
            nll = dy.pickneglogsoftmax(self.structural_W  * dy.rectify(stack_state.output())  + self.structural_b,ref_idx)
        else:
            print('error in evaluation')

        return nll

    
    def eval_sentence(self,ref_tree,backprop=True):
        """
        Evaluates the model predictions against the reference data.
        and optionally performs backpropagation. 

        Args:
          ref_tree     (ConsTree): the reference tree.
          backprop         (bool): a flag telling if we perform backprop
        Returns:
           (float,float,int,int) : the model NLL, the word only NLL, the size of the derivation, the number of predicted words 
        """
        sentence = ref_tree.tokens()
        derivation,last_config = self.static_inorder_oracle(ref_tree,sentence)

        N  = len(sentence)
        dN = len(derivation)

        all_NLL     = []
        lexical_NLL = []
    
        dy.renew_cg()
        configuration = self.init_configuration(N)

        for ref_action in derivation:

            S,B,n,stack_state,lab_state = configuration

            nll =  self.eval_action_distrib(configuration,sentence,ref_action)
            all_NLL.append( nll )

            if lab_state == RNNGparser.WORD_LABEL:
                configuration = self.generate_word(configuration,sentence)
                lexical_NLL.append(nll)
            elif lab_state == RNNGparser.NT_LABEL:
                configuration = self.label_nonterminal(configuration,ref_action)
            elif ref_action == RNNGparser.CLOSE:
                configuration = self.close_action(configuration)
            elif ref_action == RNNGparser.OPEN:
                configuration = self.open_action(configuration)
            elif ref_action == RNNGparser.SHIFT:
                configuration = self.shift_action(configuration)
            elif ref_action == RNNGparser.TERMINATE:
                pass
        
        loss     = dy.esum(all_NLL)
        lex_loss = dy.esum(lexical_NLL)

        NLL     = loss.value()
        lex_NLL = lex_loss.value()
        
        if backprop:
            loss.backward()
            self.trainer.update()
            
        return (NLL,lex_NLL,dN,N)

    
    def train_model(self,train_treebank,dev_treebank,modelname,lr=0.1,epochs=20):
        """
        Trains a full model for e epochs.
        It minimizes the NLL on the development set with SGD.

        Args:
          train_treebank (list): a list of ConsTree
          dev_treebank   (list): a list of ConsTree
          modelname    (string): the dirname of the generated model
        Kwargs:
          lr            (float): the learning rate for SGD
          epochs          (int): the number of epochs to run
        """
        
        #Trees preprocessing
        for t in train_treebank:
            ConsTree.strip_tags(t)
            ConsTree.close_unaries(t)

        for t in dev_treebank:
            ConsTree.strip_tags(t)
            ConsTree.close_unaries(t)

        #Coding & model structure
        self.code_lexicon(train_treebank)
        self.code_nonterminals(train_treebank,dev_treebank)
        self.code_struct_actions()
        self.allocate_structure()

        #Training
        self.trainer = dy.SimpleSGDTrainer(self.model,learning_rate=lr)
        min_nll      = np.inf

        for e in range(epochs):
            
            NLL,lex_NLL,N,lexN = 0,0,0,0
            for idx,tree in enumerate(train_treebank):
                loc_NLL,loc_lex_NLL,n,lex_n = self.eval_sentence(tree,backprop=True)
                NLL     += loc_NLL
                lex_NLL += loc_lex_NLL
                N       += n
                lexN    += lex_n
                sys.stdout.write('\rTree #%d'%(idx))
                
            print('\n[Training]   Epoch %d, NLL = %f, lex-NLL = %f, PPL = %f, lex-PPL = %f'%(e,NLL,lex_NLL, np.exp(NLL/N),np.exp(lex_NLL/lexN)),flush=True)
                
            NLL,lex_NLL,N,lexN = 0,0,0,0
            for idx,tree in enumerate(dev_treebank):
                loc_NLL,loc_lex_NLL,n,lex_n = self.eval_sentence(tree,backprop=False)
                NLL     += loc_NLL
                lex_NLL += loc_lex_NLL
                N       += n
                lexN    += lex_n
                
            print('\n[Validation] Epoch %d, NLL = %f, lex-NLL = %f, PPL = %f, lex-PPL = %f'%(e,NLL,lex_NLL, np.exp(NLL/N),np.exp(lex_NLL/lexN)),flush=True)
            print()
            if NLL < min_nll:
                self.save_model(modelname)

                
if __name__ == '__main__':

    train_treebank = [ ]
    train_stream   = open('ptb_train.mrg')
    for line in train_stream:
        t = ConsTree.read_tree(line)
        ConsTree.strip_tags(t)
        ConsTree.close_unaries(t)
        train_treebank.append(t)
    train_stream.close()    
    
    dev_treebank = [ ]
    dev_stream   = open('ptb_dev.mrg')
    for line in dev_stream:
        t = ConsTree.read_tree(line)
        ConsTree.strip_tags(t)
        ConsTree.close_unaries(t)
        dev_treebank.append(t)
    dev_stream.close()
     
    parser = RNNGparser('ptb-250.brown')
    parser.train_model(train_treebank,dev_treebank,'test_rnngf/test_rnngf',epoch=1)
