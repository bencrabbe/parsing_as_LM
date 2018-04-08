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
    SHIFT           = 'S'
    OPEN            = 'O'
    CLOSE           = 'C'
    TERMINATE       = 'T'

    UNKNOWN_TOKEN = '__UNK__'
    START_TOKEN   = '__START__'

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

    def oracle_derivation(self,ref_tree,root=True):
        """
        Returns an oracle derivation given a reference tree
        @param ref_tree: a ConsTree
        @return a list of (action, configuration) couples (= a derivation)
        """
        if ref_tree.is_leaf():
            return [(RNNGparser.SHIFT,ref_tree.label)]
        else:
            first_child = ref_tree.children[0]
            derivation = self.oracle_derivation(first_child,root=False)
            derivation.extend([(RNNGparser.OPEN,ref_tree.label)])
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
        for action in derivation:
            if type(action) == tuple:
                abs_action,symbol = action
                if abs_action == RNNGparser.SHIFT:
                    lex = ConsTree(symbol)
                    stack.append((lex,False))
                elif abs_action == RNNGparser.OPEN:
                    lc_child = stack.pop()
                    lc_node,status = lc_child
                    assert(status==False)
                    root = ConsTree(symbol,children=[lc_node])
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

    def code_actions(self):
        """
        Codes the actions on integers
        """
        self.actions     =   [ (RNNGparser.SHIFT,LEX) for LEX in self.rev_word_codes] 
        self.actions.extend( [ (RNNGparser.OPEN,NT) for NT in self.nonterminals ])
        self.actions.append( RNNGparser.CLOSE )
        self.actions.append( RNNGparser.TERMINATE )
        self.action_codes = dict([(s,idx) for (idx,s) in enumerate(self.actions)])
        
        #Masks
        self.open_mask      =    np.array([True]  * len(self.rev_word_codes) + [False] * len(self.nonterminals) +  [True,True])
        self.shift_mask     =    np.array([False] * len(self.rev_word_codes) + [True]  * len(self.nonterminals) +  [True,True]) 
        self.close_mask     =    np.array([True]  * len(self.rev_word_codes) + [True]  * len(self.nonterminals) +  [False,True]) 
        self.terminate_mask =    np.array([True]  * len(self.rev_word_codes) + [True]  * len(self.nonterminals) +  [True,False]) 

    #transition system
    def init_configuration(self,N):
        """
        A: configuration is a quintuple (S,B,n,stack_mem,sigma)
        
        S: is the stack
        B: the buffer
        n: the number of predicted constituents in the stack
        stack_mem: the current state of the stack lstm
        sigma:     the *local* score of the configuration. I assume scores are log probs
        
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

        return ([],tuple(range(N)),0,stackS,0.0)

    
    def shift_action(self,configuration,sentence,local_score):
        """
        That's the RNNG GENERATE/SHIFT action.
        @param configuration : a configuration triple
        @param sentence: the list of words of the sentence as a list of word idxes
        @param local_score: the local score of the action
        @return a configuration resulting from shifting the next word into the stack 
        """
        S,B,n,stack_state,score = configuration
        word_idx = sentence[B[0]]
        word_embedding = self.lex_embedding_matrix[word_idx]
        return (S + [StackSymbol(B[0],StackSymbol.COMPLETED,word_embedding)],B[1:],n,stack_state.add_input(word_embedding),local_score)

    def open_action(self,configuration,X,local_score):
        """
        That's the RNNG OPEN-X action.
        @param configuration : a configuration triple
        @param X: the category to Open
        @param local_score: the local score of the action
        @return a configuration resulting from opening the X constituent
        """
        S,B,n,stack_state,score = configuration

        nt_idx = self.nonterminals_codes[X]
        nonterminal_embedding = self.nt_embedding_matrix[nt_idx]
        stack_state = stack_state.add_input(nonterminal_embedding)

        return (S + [StackSymbol(X,StackSymbol.PREDICTED,nonterminal_embedding)],B,n+1,stack_state,local_score)

    def close_action(self,configuration,local_score):
        """
        That's the RNNG CLOSE action.
        @param configuration : a configuration triple
        @return a configuration resulting from closing the current constituent
        """
        S,B,n,stack_state,score = configuration
        assert(n > 0)
        #finds the closest predicted constituent in the stack and backtracks the stack lstm.
        midx = -1
        for idx,symbol in enumerate(reversed(S)):
            if symbol.status == StackSymbol.PREDICTED:
                midx = idx+2
                break
            else:
                stack_state = stack_state.prev()
        stack_state = stack_state.prev()
        try:
            root_symbol = S[-midx+1].copy()
            root_symbol.complete()
            children    = [S[-midx]]+S[-midx+2:]
        except:
            print(self.pretty_print_configuration(configuration))
            print(midx)
            exit(1)
            
        #compute the tree embedding with the tree_rnn
        nt_idx = self.nonterminals_codes[root_symbol.symbol]
        NT_embedding = self.nt_embedding_matrix[nt_idx]
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
        tree_embedding =  dy.tanh(W * x)
        
        return (S[:-midx]+[root_symbol],B,n-1,stack_state.add_input(tree_embedding), local_score)

    
    def next_action_mask(self,configuration,last_action,sentence):
        """
        This returns a mask stating which abstract actions are possible for next round
        @param configuration: the current configuration
        @param last_action  : the last abstract action  performed by this parser
        @param sentence     : a list of strings, the tokens
        @return a mask for the possible next actions
        """
        MASK = np.array([True] * len(self.actions))
        S,B,n,stack_state,local_score = configuration

        if not B:
            MASK *= self.open_mask
            MASK *= self.shift_mask
        if not S:
            MASK *= self.open_mask
            MASK *= self.close_mask
        if type(last_action) == tuple and last_action[0] == RNNGparser.OPEN:
            MASK *= self.open_mask
            MASK *= self.close_mask
        if n == 0:
            MASK *= self.close_mask
        if n > 0 or len(S) > 1 :
            MASK *= self.terminate_mask
        if B:
            MASK *= self.terminate_mask
            #masks for shift : only one is possible when parsing
            MASK *= self.shift_mask
            MASK [ self.action_codes[(RNNGparser.SHIFT,sentence[B[0]])] ] = 1.0
        return MASK

    
    def predict_action_distrib(self,configuration,last_action,sentence):
        """
        This predicts the next action distribution with the classifier and constrains it with the classifier structural rules
        @param configuration: the current configuration
        @param last_action  : the last action  performed by this parser
        """
        S,B,n,stack_state,local_score = configuration
        
        Wtop = dy.parameter(self.preds_out)
        Wbot = dy.parameter(self.merge_layer)
        btop = dy.parameter(self.preds_bias)
        bbot = dy.parameter(self.merge_bias)        
        probs = dy.softmax( (Wtop * dy.tanh((Wbot *stack_state.output()) + bbot)) + btop)
        return np.maximum(probs.npvalue(),np.finfo(float).eps) * self.next_action_mask(configuration,last_action,sentence)
        #this last line attempts to address numerical undeflows (0 out of dynet softmaxes) and applies the hard constraint mask
        #such that a legal action has a prob > 0.
    
    def train_one(self,configuration,ref_action):
        """
        This performs a forward, backward and update pass on the network for this action.
        @param configuration: the current configuration
        @param ref_action  : the reference action
        @return the loss for this action
        """
        S,B,n,stack_state,local_score = configuration
        
        Wtop   = dy.parameter(self.preds_out)
        Wbot   = dy.parameter(self.merge_layer)
        btop   = dy.parameter(self.preds_bias)
        bbot   = dy.parameter(self.merge_bias)
        probs  = dy.softmax( (Wtop * dy.tanh((Wbot * stack_state.output()) + bbot)) + btop)
        loss   = dy.pickneglogsoftmax(probs,self.action_codes[ref_action])
        loss_val = loss.value()
        loss.backward()
        self.trainer.update()
        return loss_val

    def beam_parse(self,tokens,all_beam_size,lex_beam_size,ref_tree=None):
        """
        This parses a sentence with word sync beam search.
        The beam search assumes the number of structural actions between two words to be bounded 
        @param tokens: the sentence tokens
        @param ref_tree: if provided return an eval against ref_tree rather than a parse tree
        @return a derivation, a ConsTree or some evaluation metrics
        """
        class BeamItem:
            def __init__(self,prev_item,prev_action,config,prefix_score):
                self.prev        = prev_item #prev beam item
                self.pred_action = prev_action
                self.config      = config
                self.score       = prefix_score
                
        dy.renew_cg()
        tokens    = [self.lex_lookup(t) for t in tokens  ]
        tok_codes = [self.word_codes[t] for t in tokens  ]    
        C         = self.init_configuration(len(tokens))
        all_beam  = [ BeamItem(None,'init',C,0) ]
        next_lex_beam = [ ]
        for idx in range(len(tokens) + 1):
            while all_beam:
                next_all_beam = []
                for elt in all_beam:
                    C = elt.config
                    s = elt.score
                    prev_action = elt.pred_action
                    probs = np.log(self.predict_action_distrib(C,prev_action,tokens))
                    for act,logprob in zip(self.actions,probs):
                        if logprob > -np.inf:#filters illegal actions
                            if act ==  RNNGparser.TERMINATE:
                                next_lex_beam.append( BeamItem(elt,act,None,s+logprob) )
                            elif type(act) == tuple and act[0] == RNNGparser.SHIFT:
                                next_lex_beam.append( BeamItem(elt,act,None,s+logprob) )
                            else: #not a shift
                                next_all_beam.append( BeamItem(elt,act,None,s+logprob) )
                #prune and exec actions
                next_all_beam.sort(key=lambda x:x.score,reverse=True)
                next_all_beam = next_all_beam[:all_beam_size]
                for elt in next_all_beam:
                    C   = elt.prev.config
                    act = elt.pred_action
                    if act == RNNGparser.CLOSE:
                        elt.config = self.close_action(C,0)
                    elif act == RNNGparser.TERMINATE:
                        elt.config = C
                    elif act[0] == RNNGparser.SHIFT:
                        elt.config = self.shift_action(C,tok_codes,0)
                    elif act[0] == RNNGparser.OPEN:
                        elt.config = self.open_action(C,act[1],0)
                all_beam = next_all_beam
            #Lex beam
            next_lex_beam.sort(key=lambda x:x.score,reverse=True)
            next_lex_beam = next_lex_beam[:lex_beam_size]
            for elt in next_lex_beam:
                C = elt.prev.config
                act = elt.pred_action
                if act == RNNGparser.TERMINATE:
                    elt.config = C
                else:
                    elt.config = self.shift_action(C,tok_codes,0)
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
        pred_action = 'init'
        S,B,n,stackS,score = C
        deriv = [ ]
        #while B or len(S) > 1 or n != 0:
        while True:
            probs = self.predict_action_distrib(C,pred_action,tokens)
            max_idx   = np.argmax(probs)
            score = probs[max_idx]
            pred_action = self.actions[max_idx]
            deriv.append(pred_action)            
            if pred_action == RNNGparser.CLOSE:
                C = self.close_action(C,score)
            elif pred_action == RNNGparser.TERMINATE: #we exit the loop here
                break #  <= EXIT
            elif pred_action[0] == RNNGparser.SHIFT:
                print(C,tok_codes)
                C = self.shift_action(C,tok_codes,score)
            elif pred_action[0] == RNNGparser.OPEN:
                C = self.open_action(C,pred_action[1],score)
            S,B,n,stackS,score = C
            if len(deriv) > 5000: #useful in case of numerical underflow
                print(deriv)
                exit(1)
        if get_derivation:
            return deriv
        #print('DERIV',deriv)
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
        
        #top level MLP
        self.preds_out             = self.model.add_parameters((actions_size,self.hidden_size),init='glorot')          #action output layer
        self.preds_bias            = self.model.add_parameters((actions_size),init='glorot')
        self.merge_layer           = self.model.add_parameters((self.hidden_size,self.stack_hidden_size),init='glorot')
        self.merge_bias            = self.model.add_parameters((self.hidden_size),init='glorot')
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

        
    def eval_model(self,ref_treebank,all_beam_size,lex_beam_size):
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


                    
    def train_generative_model(self,max_epochs,train_bank,dev_bank,lex_embeddings_file=None,learning_rate=0.001):
        """
        This trains an RNNG model on a treebank
        @param learning_rate: the learning rate for SGD
        @param max_epochs: the max number of epochs
        @param train_bank: a list of ConsTree
        @param dev_bank  : a list of ConsTree
        @param lex_embeddings_file: an external word embeddings filename
        @return a dynet model
        """
        #Coding
        self.code_lexicon(train_bank,self.max_vocab_size)
        self.code_nonterminals(train_bank)
        self.code_actions()

        self.print_summary()
        print('---------------------------')
        print('num epochs          :',max_epochs)
        print('learning rate       :',learning_rate)        
        print('num training trees  :',len(train_bank))

        self.make_structure()
        
        #training
        self.trainer = dy.AdamTrainer(self.model,alpha=learning_rate)
        for e in range(max_epochs):
            loss,N = 0,0
            for tree in train_bank:
                dy.renew_cg()
                ref_derivation  = self.oracle_derivation(tree)
                #print(ref_derivation)
                tokens    = [self.lex_lookup(t) for t in tree.tokens()  ]
                tok_codes = [self.word_codes[t] for t in tokens  ]   
                step, max_step  = (0,len(ref_derivation))
                current_config  = self.init_configuration(len(tokens))
                while step < max_step:
                    ref_action = ref_derivation[step]
                    #print(ref_action)
                    loss += self.train_one(current_config,ref_action)
                    N    += 1
                    if ref_action == RNNGparser.CLOSE:
                        current_config = self.close_action(current_config,0.0)
                    elif ref_action[0] == RNNGparser.SHIFT:
                        current_config = self.shift_action(current_config,tok_codes,0.0)
                    elif ref_action[0] == RNNGparser.OPEN:
                        current_config = self.open_action(current_config,ref_action[1],0.0)
                    step += 1
                    #print(self.pretty_print_configuration(current_config))
            sys.stdout.write("\rEpoch %d, Mean Loss : %.5f"%(e,loss/N))
            sys.stdout.flush()
        print()



        
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
            
    train_treebank = []

    if train_file and model_name:
        train_treebank = []
        train_stream   = open(train_file)
        for line in train_stream:
            train_treebank.append(ConsTree.read_tree(line))
        p = RNNGparser(max_vocabulary_size=TrainingParams.LEX_MAX_SIZE,\
                        hidden_size=StructParams.OUTER_HIDDEN_SIZE,\
                        stack_embedding_size=StructParams.STACK_EMB_SIZE,\
                        stack_memory_size=StructParams.STACK_EMB_SIZE)
        p.train_generative_model(TrainingParams.NUM_EPOCHS,train_treebank,[],learning_rate=TrainingParams.LEARNING_RATE)
        p.save_model(model_name)
        train_stream.close()
        
    if model_name and raw_file:
        p = RNNGparser.load_model(model_name)
        test_stream = open(raw_file)
        for line in test_stream:
            print(p.parse_sentence(line.split(),ref_tree=None))
            print(p.beam_parse(line.split(),all_beam_size=64,lex_beam_size=8))
        test_stream.close()
            
    #t  = ConsTree.read_tree('(S (NP Le chat ) (VP mange  (NP la souris)))')
    #t2 = ConsTree.read_tree('(S (NP Le chat ) (VP voit  (NP le chien) (PP sur (NP le paillasson))))')
    #t3 = ConsTree.read_tree('(S (NP La souris (Srel qui (VP dort (PP sur (NP le paillasson))))) (VP sera mangée (PP par (NP le chat ))))')

    #p = RNNGparser(hidden_size=50,stack_embedding_size=50,stack_memory_size=25)
    #p.train_generative_model(500,[t,t2,t3],[])
    #D = p.oracle_derivation(t2)
    #print(D)
    #print(RNNGparser.derivation2tree(D))
    #print(p.parse_sentence(t.tokens(labels=True),ref_tree=None))
    #print(p.beam_parse(t.tokens(labels=True),all_beam_size=64,lex_beam_size=8))

    #print(p.parse_sentence(t2.tokens(labels=True),ref_tree=None))
    #print(p.beam_parse(t2.tokens(labels=True),all_beam_size=64,lex_beam_size=8))

    #print(p.parse_sentence(t3.tokens(labels=True),ref_tree=None))
    #print(p.beam_parse(t3.tokens(labels=True),all_beam_size=64,lex_beam_size=8))
    
    #print()
    #print(p.oracle_derivation(t2))
    #print(p.parse_sentence(t2.tokens(labels=True),ref_tree=None))
    #print(p.parse_sentence(t3.tokens(labels=True),ref_tree=None))
