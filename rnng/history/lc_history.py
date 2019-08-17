import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from constree    import *
from collections import Counter
from random      import shuffle,random
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

#MODELS
#Derivation = w1 s1 w2 s2   ...   wN sN
#Model 1        
#  P(wi | w1 s1 ... wi-1  si-1)
#Estimation
#    ~ P(w_i | w1 ... wi-1 ) [ task 1 : language model where structure is only used at estimation]
#      P(s_i | w1 ... w_i  ) [ task 2 : predict annotations outside language model. ]

#with the additional independance assumptions we have :
#P(w_i | w1 ... wi-1 ) decomposes as:
#   P(wordlabel_i | w1 ... wi-1 ) P(wordaction_i | w1 ... wi-1)
#P(s_i | w1 ... wi ) decomposes as:
#   P(NTlabel_i | w1 ... wi ) P(NTaction_i | w1 ... wi)

#The key property of the model is that it can learn from unsupervised data the language model subtask with  
#   >>> P(wordlabel_i | w1 ... wi-1 ) <<<

# Prediction
# these substasks may use constraints that create a conditioning on previous hidden struct too -> not all struct actions are always possible
#use beam search
# Exact decomposition :
# prod_i^N  P(wi | w1,s1 ...  wi-1,si-1) P(si | (w1,s1) ... (wi-1,si-1), wi)
# Our approximation (in exchange for cheap unsup learning)  :
# prod_i^N  P(wi | w1 ...  wi-1)         P(si | w1 ... wi-1, wi)
# We add an additional conditional independance assumption to isolate the language model :
# xxx

#Model 2
#  P(ai | a1 ...  ai-1) -> requires true beam search because si gets back on input like in seq2seq
#                          (for training it is not sure that it changes a lot, requires to input the struct actions but that's it)
#  unsupervised = xxx boils down to basic domain adaptation   

 
class LCtree (ConsTree):
    """
    Specialization of ConsTree for coding complete/incomplete constituents as used in this transition system.
    """
    def __init__(self,label,children=None):
        super(LCtree, self).__init__(label,children)
    
    def rightmost_attach_site(self):
        """
        Returns the rightmost node in a constituent tree
        to which we can attach a right child. 
        Args:
             ctree (ConsTree): tree where to search
        Returns:
              the immediate root of the attachement site if such node exists else None
        """
        if self.arity() == 2:
            return self.get_child(1).rightmost_attach_site()
        else: #arity < 2
            return None if self.is_leaf() else self  

    def is_complete(self):
        return self.rightmost_attach_site() is None
    

#DATA SET
#This recodes with more flexibility some key functionalities for managing data inspired by
#torch.text by making them suitable for use for parsing purposes.
    
class Vocabulary:

    def __init__(self,counter,min_freq=1,pad='<pad>',unk=None,sos=None,vectors=None):
        """
        New vocabulary.
        Args:
            counter (collections.Counter): tokens frequencies
            min_freq                (int): words with counts strictly less than min_freq are removed from  the vocabulary 
            unk                  (string): the reserved 'unk' token
            pad                  (string): the reserved 'pad' token
            sos                  (string): the reserved start of sentence token
            vectors              (string): filename where to load pretrained embeddings vectors
        """
        self.unk = None
        
        self.counts = Counter(dict([ (item,count) for (item,count) in counter.items() if count >= min_freq] ))
        self.itos   = [ ]

        if unk and not unk in self.counts:
            self.itos.append(unk)
            self.unk = unk
        if pad and pad not in self.counts:
            self.itos.append(pad)
            self.pad = pad
        if sos and sos not in self.counts:
            self.itos.append(sos)
            self.sos = sos
    
        self.itos.extend( self.counts.keys() )    
        self.stoi = dict(zip(self.itos,range(len(self.itos))))
        self.vectors = vectors

    def size(self):
        """
        Returns the number of symbols in this vocabulary
        """
        return len(self.itos)
    
    def token_index(self,token):
        """
        Gets the index of a token
        Args: 
           token  (str): the token for which we seek the index
        Returns:
           the index of the token, the index of the unk if not found and an error if unk is not in this dict
        """
        return self.stoi.get(token,0) if self.unk else self.stoi[token]

    def token_form(self,tok_idx):
        """
        Retrieves a token string from its index.
        Note that if the token was unknown to the vocabulary, the orginal form is not retrieved (rather the unk code)
        Args:
           tok_idx (int): the token integer index
        Returns:
           a string.
        """
        return self.itos[tok_idx] 
    
class ParsingDataSet(object): 
    """
    That's a data set for parsing. Each example is a couple made of a list of tokens and an optional derivation.
    That's currently tied to the parser class (try to remove this dependency later)
    """
    def __init__(self,dataset,ext_vocab=None,root_dataset=None, unk='<unk>',pad='<pad>',sos='<sos>',min_lex_counts=0):
        """
        Args:
             dataset               (list): a list of trees (or a list of strings for test and unsupervised setups)
             ext_vocab       (Vocabulary): if specified use existing external vocabulary, otherwise builds it from the data
             root_dataset(ParsingDataSet): if specified uses the vocabulary encoding from this external dataset rather than inferring encodings from this one. 
             unk                    (str): a string for the unk token for internal vocab
             pad                    (str): a string for the pad token for internal vocab
             sos                    (str): a string for the start of sentence (sos) token
             min_lex_counts         (int): count threshold under which tokens are excluded from the lexical vocabulary built from the data
        """

        is_treebank = isinstance(dataset[0],ConsTree)
        
        #1. Data structuration 
        if is_treebank : #has annotated trees
            
            self.tree_set      = dataset
            derivations        = [ LCmodel.oracle_derivation(tree) for tree in dataset ]
                
            self.tokens        = [ tree.tokens() for tree in dataset ]
            self.lex_actions   = [ self.extract_lex_actions(deriv)  for deriv in derivations ]
            self.struct_labels = [ self.extract_struct_labels(deriv)  for deriv in derivations ]
            self.struct_actions= [ self.extract_struct_actions(deriv)  for deriv in derivations ]
    
        else:                             #raw text
            self.tokens = [ sent.split( ) for sent in dataset ]

        #2. Vocabularies
        if root_dataset :
            self.lex_vocab = root_dataset.lex_vocab
            if is_treebank :
                self.lex_action_vocab    = root_dataset.lex_action_vocab   
                self.struct_vocab        = root_dataset.struct_vocab
                self.struct_action_vocab = root_dataset.struct_action_vocab
            self.unk = root_dataset.unk
            self.pad = root_dataset.pad
            self.sos = root_dataset.sos       
        else:
            self.lex_vocab               = ParsingDataSet.build_vocab(self.tokens,unk_lex=unk,pad=pad,sos=sos,min_counts=min_lex_counts)
            if is_treebank :
                self.lex_action_vocab    = ParsingDataSet.build_vocab(self.lex_actions,pad=pad)
                self.struct_vocab        = ParsingDataSet.build_vocab(self.struct_labels,pad=pad,sos=sos)
                self.struct_action_vocab = ParsingDataSet.build_vocab(self.struct_actions,pad=pad,sos=sos)
            self.unk = unk
            self.pad = pad
            self.sos = sos
            
        #3. External vocabulary
        if ext_vocab:
            assert(ext_vocab.unk == unk and ext_vocab.sos == sos and ext_vocab.pad == pad)
            self.lex_vocab = ext_vocab
            
    def decode_derivation(self,derivation): #pred_lexaction,pred_ytokens,pred_structaction,pred_structlabels
        """
        This translates an integer coded derivation back to a string.
        Args :
           xtokens    (list): list of integers (token codes)
           derivation (list): list of integer tuples
        """
        def translate(action,label,idx):
            if idx % 2 == 0 :
                return (self.lex_action_vocab.itos[action],self.lex_vocab.itos[label])
            else:
                return (self.struct_action_vocab.itos[action],self.struct_vocab.itos[label])
            
        return [translate(action,label,idx) for idx, (action,label) in enumerate(derivation) ] 
            
    def is_training_set(self):
        """
        Returns true if this dataset has annotated trees
        """
        return hasattr(self,'lex_actions')

    @staticmethod
    def build_vocab(datalist,unk_lex=None,pad=None,sos=None,min_counts=0):
        """
        Builds a vocabulary from a matrix of strings 
        Args: 
           datalist (list of list) : a list of ConsTree objects 
           unk_lex           (str) : a string to use for replacement of unknown words
           pad               (str) : a string to use padding tokens
           sos               (str) : a string to use start of sentence tokens
        Returns 
            a Vocabulary object.
        """
        token_counter    = Counter()
        for item in datalist:
            token_counter.update(item)
        return Vocabulary(token_counter,unk=unk_lex,pad=pad,sos=sos,min_freq=min_counts)
    
    def extract_tokens(self,derivation):
        """
        Extracts the list of words from a derivation and returns it
        Args: 
            derivation (list): a list of couples (action,label)
        Return :
            list. A list of word tokens
        """        
        return [ label for (idx,(action,label)) in enumerate(derivation) if idx % 2 == 0]

    def extract_lex_actions(self,derivation):
        """
        Extracts the list of lexical actions from a derivation and returns it
        Args: 
            derivation (list): a list of couples (action,label)
        Return :
            list. A list of word tokens
        """        
        return [ action for (idx,(action,label)) in enumerate(derivation) if idx % 2 == 0]

    def extract_struct_labels(self,derivation):
        """
        Extracts the list of non terminal symbols from a derivation and returns it
        Args: 
            derivation (list): a list of couples (action,label)
        Return :
            list. A list of non terminal symbols
        """                
        return [ label for (idx,(action,label)) in enumerate(derivation) if idx % 2 == 1]

    def extract_struct_actions(self,derivation):
        """
        Extracts the list of structural actions from a derivation and returns it
        Args: 
            derivation (list): a list of couples (action,label)
        Return :
            list. A list of structural actions
        """                
        return [ action for (idx,(action,label)) in enumerate(derivation) if idx % 2 == 1]

    def __len__(self):
        """
        Returns the number of samples in the data set
        """
        return len(self.tokens)

    def example_length(self,idx):
        """
        Returns the length of an example (token length for batch sorting purposes)
        """
        return len(self.tokens[idx])
    
    def sample_tokens(self,toklist,batch_len,alpha=0.0):
        """ 
        This returns a list of integers encoding the tokens
        Args:
            toklist         (list): a list of strings
            alpha          (float): alpha >= 0 a constant used for sampling unk words. if alpha == 0, no sampling occurs
        """
        def sample_unk(token,alpha):
            prob = random()
            C = self.lex_vocab.counts[token]
            if C == 0 or prob < alpha / C:
                return self.unk 
            return token

        def subst_unk(token): 
            C = self.lex_vocab.counts[token]
            if C == 0:
                return self.unk
            return token
        
        if alpha > 0:
            return [sample_unk(tok,alpha) for tok in toklist ]
        else:
            return [subst_unk(tok) for tok in toklist ]
            
 
    def numericalize_example(self,datum,batch_len,vocabulary):
        """
        This returns a list of integers encoding the given symbols. 
        Args:
            datum           (list): list of strings, the tokens to map to integers
            batch_len        (int): the length of the batch (>= actual number of tokens). Missing actions are padded left
            vocabulary(Vocabulary): a vocabulary object that performs the tokenwise mapping 
        Returns:
            A list of integers
        """    
        N                 = len(datum)
        padded_datum      = datum + [self.pad]*(batch_len-N)
        return [vocabulary.token_index(token) for token in padded_datum]
 
    
class ParseBatch: 
   
    def __init__(self,xtokens=None,ytokens=None,struct_actions=None,struct_labels=None,lex_actions=None,lex_labels=None,token_length=None,orig_idxes=None,derivation_length=None):
        self.xtokens           = xtokens
        self.ytokens           = ytokens
        self.lex_actions       = lex_actions
        self.lex_labels        = lex_labels
        self.struct_actions    = struct_actions
        self.struct_labels     = struct_labels
        self.tokens_length     = token_length
        self.orig_idxes        = orig_idxes
         
class BucketLoader:
    """
    This BucketLoader is an iterator that reformulates torch.text.data.BucketIterator
    """
    def __init__(self,dataset,batch_size,device=-1,alpha=0.0): 
        """
        This function is responsible for delivering batches of examples from the dataset for a given epoch.
        It attempts to bucket examples of the same size in the same batches.

        Args:
          dataset   (ParsingDataset): the treebank from where to get samples
          batch_size           (int): the size of the batch
          device               (str): a string that says on which device the batched data should live (defaults to cpu).
          alpha              (float): param for sampling unk words
        """ 
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.device      = device
        self.current_idx = 0
        self.alpha       = alpha
        
        self.data_idxes  = list(range(len(self.dataset)))

        
    def encode_batch(self,batch_idxes):
        """
        Encode and adds padding symbols to shorter examples in this batch
        Args:
            batch_idxes (list): a list of integers, indexes of the examples included in this batch
        Returns:
            a ParseBatch. The batch object fields are filled partly if not supervised training set 
            The batches examples are padded right.
        """
        batchN             = len(batch_idxes)
        token_lengths      = [ len(self.dataset.tokens[idx]) for idx in batch_idxes ]
        max_token_length   = max(token_lengths)
        
        raw_tokens    = [ self.dataset.sample_tokens(self.dataset.tokens[batch_idxes[step]],max_token_length,alpha=self.alpha) for step in range(batchN) ]
        ytoken_matrix = [ self.dataset.numericalize_example(elt,max_token_length,self.dataset.lex_vocab) for elt in raw_tokens ]
        xtoken_matrix = [ self.dataset.numericalize_example([self.dataset.sos]+elt[:-1],max_token_length,self.dataset.lex_vocab) for elt in raw_tokens ]

        xtoken_tensor  = torch.tensor(xtoken_matrix,dtype=torch.long,device=self.device)
        ytoken_tensor  = torch.tensor(ytoken_matrix,dtype=torch.long,device=self.device)

        print(raw_tokens)

        
        if self.dataset.is_training_set():
            
            lex_action_matrix     = [self.dataset.numericalize_example( self.dataset.lex_actions[batch_idxes[step]], max_token_length,self.dataset.lex_action_vocab)  for step in range(batchN) ]
            struct_action_matrix  = [self.dataset.numericalize_example([self.dataset.sos]+self.dataset.struct_actions[batch_idxes[step]], max_token_length,self.dataset.struct_action_vocab) for step in range(batchN) ]
            struct_label_matrix   = [self.dataset.numericalize_example([self.dataset.sos]+self.dataset.struct_labels[batch_idxes[step]], max_token_length,self.dataset.struct_vocab) for step in range(batchN) ]
                        
            lex_action_tensor     = torch.tensor(lex_action_matrix,dtype=torch.long,device=self.device)
            struct_action_tensor  = torch.tensor(struct_action_matrix,dtype=torch.long,device=self.device)
            struct_label_tensor   = torch.tensor(struct_label_matrix,dtype=torch.long,device=self.device)
        
            return ParseBatch(xtokens=xtoken_tensor,ytokens=ytoken_tensor, lex_actions=lex_action_tensor,\
                                  struct_actions=struct_action_tensor,struct_labels=struct_label_tensor,token_length=token_lengths,orig_idxes=batch_idxes)
        
        return ParseBatch(xtokens=xtoken_tensor,ytokens=ytoken_tensor,token_length=token_lengths,orig_idxes=batch_idxes) 
            
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        This yields a batch of data.
        """
        if self.current_idx == 0 :#init epoch
            shuffle(self.data_idxes)
            lengths         = [ self.dataset.example_length(idx) for idx in self.data_idxes ]
            self.data_idxes = [idx for (idx,length) in sorted(zip(self.data_idxes,lengths),key=lambda x:x[1],reverse=True) ]

        if self.current_idx < len(self.data_idxes):            
            batch_idxes  = self.data_idxes[ self.current_idx:self.current_idx+self.batch_size ]
            self.current_idx += self.batch_size
            return self.encode_batch(batch_idxes)
        else:
            raise StopIteration
        pass #return

          
class LCmodel(nn.Module):
    
    ACTION_SHIFT_INIT    = 'I' 
    ACTION_SHIFT_ATTACH  = 'S' 
    ACTION_ATTACH        = 'A' 
    ACTION_PREDICT       = 'P' 

    def __init__(self,ref_set,rnn_memory=100,embedding_size=100,device=-1):
        """
        This allocates the model parameters on the machine.
        Args:
           ref_set (ParsingDataSet): the reference training from which vocabularies are built
           rnn_memory  (int)  : the size of the rnn hidden memory 
           embedding_size(int): the embedding size
           device        (int): the device where to store the params (-1 :cpu ; 0,1,2... : GPU identifier)
        """
        super(LCmodel, self).__init__()
        self.ref_set        = ref_set
        self.rnn_memory     = rnn_memory
        self.embedding_size = embedding_size
        self.allocate_structure(device)
        
    def allocate_structure(self,device=-1):
        """
        This allocates the model parameters on the machine.
        Args:
           action_size   (int): the number of action types
           lex_size      (int): the size of the lexicon vocabulary
           struct_size   (int): the size of the non terminal vocabulary
           rnn_memory    (int): the size of the rnn hidden memory 
           embedding_size(int): the embedding size
           device        (int): the device where to store the params (-1 :cpu ; 0,1,2... : GPU identifier)
        """
        self.E               = nn.Embedding( self.ref_set.lex_vocab.size(),self.embedding_size)
        self.lstm            = nn.LSTM(self.embedding_size, self.rnn_memory,num_layers=1,bidirectional=False)
        
        self.W_struct_label  = nn.Linear(self.rnn_memory, self.ref_set.struct_vocab.size())     
        self.W_lex_label     = nn.Linear(self.rnn_memory, self.ref_set.lex_vocab.size())    
        self.W_lex_action    = nn.Linear(self.rnn_memory, self.ref_set.lex_action_vocab.size()) 
        self.W_struct_action = nn.Linear(self.rnn_memory, self.ref_set.struct_action_vocab.size())    
        self.softmax         = nn.LogSoftmax(dim=1)

    def forward_lexical_actions(self,base_output):
        """
        Performs the forward pass for the lexical action subtask
        Args:
            base_output  (tensor): the tensor outputted by the base LSTM encoder.
        Returns:
            a tensor. A list of softmaxed predictions for each example provided as argument.
        """
        return self.softmax(self.W_lex_action(base_output))    

    def forward_lexical_tokens(self,base_output):
        """
        Performs the forward pass for the lexical token subtask
        Args:
            base_output  (tensor): the tensor outputted by the base LSTM encoder.
        Returns:
            a tensor. A list of softmaxed word predictions for each example provided as argument.
        """
        #  @see AdaptiveLogSoftmaxWithLoss in pytorch + requirements (sorting etc.)
        #  @see https://towardsdatascience.com/speed-up-your-deep-learning-language-model-up-to-1000-with-the-adaptive-softmax-part-1-e7cc1f89fcc9
        return self.softmax(self.W_lex_label(base_output))    
    
    def forward_structural_actions(self,base_output):
        """
        Performs the forward pass for the structural actions subtask
        Args:
            base_output  (tensor): the tensor outputted by the base LSTM encoder.
        Returns:
            a tensor. A list of softmaxed actions predictions for each example provided as argument.
        """
        return self.softmax(self.W_struct_action(base_output))

    def forward_structural_labels(self,base_output):
        """
        Performs the forward pass for the structural labels subtask
        Args:
            base_output  (tensor): the tensor outputted by the base LSTM encoder.
        Returns:
            a tensor. A list of softmaxed non terminal predictions for each example provided as argument.
        """
        return self.softmax(self.W_struct_label(base_output)) 
    
    def forward_base(self,xinput,true_batch_lengths,train_mode=True):
        """
        Args :
           xinput           (tensor): an integer coded input 
           true_batch_lengths (list): list of integers (true lengths of inputs)
        Returns: 
           The RNN output as a pytorch flattened Sequence of wordwise encodings
        """
        #@see (packing) https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
        xembedded         = self.E(xinput)                                                        #xembedded [dim] = batch_size x sent_len x embedding_size
        xembedded         = pack_padded_sequence(xembedded, true_batch_lengths, batch_first=True)
        lstm_out, _       = self.lstm(xembedded)                                                 
        lstm_out, _       = pad_packed_sequence(lstm_out,batch_first=True)                        #lstm_out  [dim] = batch_size x sent_len x hidden_size
          
        # We flatten the batch before applying the linear outputs (required by the loss functions at the end)
        batch_size,sent_len,hidden_size = lstm_out.shape
        #Reshapes the output as a flat sequence compatible with softmax and loss functions
        lstm_out = lstm_out.contiguous().view(batch_size*sent_len,hidden_size)                    #lstm_out  [dim] = (batch_size*sent_len) x hidden_size
        return lstm_out
        
    def decode(self,tokens,pred_lexaction,pred_ytokens,pred_structaction,pred_structlabels,true_length):
        """
        This decodes a batched prediction into a tree. (~ performs best first parsing)
        Args:
           tokens           (tensor) : a list of integers (the true input tokens)
           pred_lexaction    (tensor) : a word sync tensor encoding lexical actions
           pred_ytokens      (tensor) : a word sync tensor encoding lexical labels
           pred_structaction (tensor) : a word sync tensor encoding structural actions
           pred_structlabels (tensor) : a word sync tensor encoding structural labels
           true_length          (int) : true length of the input (excludes padding)
        Returns.
           A couple. A parse derivation and a parse tree (or None if failure ?)
        """
        def decode_structural(nonterminal_label,nonterminal_action):
            """
            Converts back integer codes to strings for the structural case
            """
            action = self.ref_set.struct_action_vocab.itos[nonterminal_action]
            label  = self.ref_set.struct_vocab.itos[nonterminal_label]
            return label,action

        def decode_lexical(token,lexaction=None):
            """
            Converts back integer codes to strings for the lexical case
            """
            label  = self.ref_set.lex_vocab.itos[token]
            if lexaction:
                action = self.ref_set.lex_action_vocab.itos[lexaction]
                return label,action
            return label
            
        lextokens     = pred_ytokens.cpu().numpy() #use torch.no_grad before parsing code and remove the detach call
        lexactions    = pred_lexaction.cpu().numpy()
        structlabs    = pred_structlabels.cpu().numpy()
        structactions = pred_structaction.cpu().numpy()
        
        shift_init_c   = self.ref_set.lex_action_vocab.token_index( LCmodel.ACTION_SHIFT_INIT )
        shift_attach_c = self.ref_set.lex_action_vocab.token_index( LCmodel.ACTION_SHIFT_ATTACH )
        predict_c      = self.ref_set.struct_action_vocab.token_index( LCmodel.ACTION_PREDICT )
        attach_c       = self.ref_set.struct_action_vocab.token_index( LCmodel.ACTION_ATTACH )
        struct_pad_c   = self.ref_set.struct_action_vocab.token_index( self.ref_set.pad )
        struct_sos_c   = self.ref_set.struct_action_vocab.token_index( self.ref_set.sos )
        lex_pad_c      = self.ref_set.lex_action_vocab.token_index( self.ref_set.pad )
        
        Stack,Buffer  = [  ] , tokens.cpu().numpy()
        derivation    = [  ]
        logprob       = 0.0
        
        N = true_length
        r,d = 0,0
        for (idx,token,laction,ntlabel,ntaction) in zip(range(N),lextokens,lexactions,structlabs,structactions):
            #STRUCTURAL STATE
            d = len(Stack) if Stack else 0  #stack depth
            r = N - 1 - idx           #remaining words
            if idx > 0 : #skips start of sentence dummy state
                #preconditions (we mask for log probs)
                ntaction[ struct_pad_c ] = np.NINF
                ntaction[ struct_sos_c]  = np.NINF
                if d > 0 and not Stack[-1].is_complete():
                    ntaction[ predict_c ] = np.NINF
                if not d <= r+1:
                    ntaction[ predict_c ]  = np.NINF
                if d == 1:
                    ntaction[ attach_c ]  = np.NINF
                #decision
                #print('ntlabel',list(zip(self.ref_set.struct_vocab.itos,np.exp(ntlabel))))
                ntlabel,struct_action = decode_structural(np.argmax(ntlabel),np.argmax(ntaction))
                #exec
                if struct_action ==  LCmodel.ACTION_PREDICT :
                    Stack[-1] = LCtree(ntlabel,children=[Stack[-1]])
                elif struct_action == LCmodel.ACTION_ATTACH :
                    rc = Stack[-2].rightmost_attach_site()
                    rc.add_child(LCtree(ntlabel,children=[Stack[-1]]))
                    Stack.pop() 
                else:
                    print('structural action problem',struct_action)
                derivation.append( (struct_action,ntlabel) )
            #LEXICAL STATE
            d = len(Stack) if Stack else 0  #stack depth
            #preconditions
            laction[ lex_pad_c ] = np.NINF
            if d == 0 :
                laction[ shift_attach_c ] = np.NINF
            elif d > 0 and Stack[-1].is_complete():
                laction[ shift_init_c ] = np.NINF
            elif not d < r:
                laction[ shift_init_c ] = np.NINF
            #decision 
            ytoken,lex_action = decode_lexical(np.argmax(token), np.argmax(laction)) #pick the relevant prob for the token here ! (to be reworked)
            #exec
            b0 = decode_lexical(Buffer[idx]) #forces the xtoken to be the reference rather than the predicted one
            if lex_action ==  LCmodel.ACTION_SHIFT_INIT:
                b0 = decode_lexical(Buffer[idx]) 
                Stack.append( LCtree(b0) )
            elif lex_action == LCmodel.ACTION_SHIFT_ATTACH :
                rc = Stack[-1].rightmost_attach_site()
                rc.add_child( LCtree(b0) )
            else:
                print('lexical action problem',struct_action)
            derivation.append( (lex_action,b0) )

        #print("Derivation",derivation)
        print("stack tree",Stack[-1])
        return derivation, Stack[-1]

    def predict(self,dev_set,batch_size=1,device=-1): 
        """
        Evaluates the parser on a dev set.
        Args:
           dev_set (ParsingDataSet): the development set
           batch_size         (int): the size of the batch
        Returns:
           a list of trees. Returns the predicted data set as whole. The original ordering of dev_set is guaranteed to be preserved.
        """
        with torch.no_grad():
            
            dataloader = BucketLoader(dev_set,batch_size,device)
            orig_idxes = [ ]
            pred_trees = [ ]

            print('predict')

            
            for batch in dataloader:
                print('predict xtokens',batch.xtokens)
                seq_representation =  self.forward_base(batch.xtokens,batch.tokens_length)
                 
                pred_lexaction     =  self.forward_lexical_actions(seq_representation)
                pred_structaction  =  self.forward_structural_actions(seq_representation)
                pred_ytokens       =  self.forward_lexical_tokens(seq_representation)
                pred_structlabels  =  self.forward_structural_labels(seq_representation)

                print('predict slabels',pred_structlabels)

                
                #here we reshape sentence_wise :
                #   input  [dim] = (batch_size*sent_len) x hidden_size
                batch_size,batch_len = batch.ytokens.shape
                #print('eval batch size x batch len (', batch_size,batch_len,')')
                #   output [dim] = batch_size x sent_len x hidden_size

                pred_lexaction    = pred_lexaction.view(batch_size,batch_len,-1)   
                pred_structaction = pred_structaction.view(batch_size,batch_len,-1)   
                pred_ytokens      = pred_ytokens.view(batch_size,batch_len,-1)   
                pred_structlabels = pred_structlabels.view(batch_size,batch_len,-1)   

                orig_idxes.extend(batch.orig_idxes)
                batch_preds = [ self.decode(batch.ytokens[idx],\
                                            pred_lexaction[idx],\
                                            pred_ytokens[idx],\
                                            pred_structaction[idx],\
                                            pred_structlabels[idx],\
                                            batch.tokens_length[idx]) for idx in range(batch_size)]

                pred_trees.extend(batch_preds)
                
            matched_idxes = enumerate(orig_idxes) #iterates using the ascending original order of the data set                    
            return [ pred_trees[current_idx] for (current_idx,orig_idx) in sorted(matched_idxes,key=lambda x:x[1]) ]

    def train(self,train_set,dev_set,epochs,raw_loader=None,batch_size=1,learning_rate=0.1,device=-1,alpha=0.0):
        """
        Args :    
          train_set (ParsingDataSet): xxx
          dev_set   (ParsingDataSet): xxx
          epochs               (int): xxx
        """
        lex_action_loss    = nn.NLLLoss()#ignore_index=train_set.lex_action_vocab.stoi[train_set.pad])
        struct_action_loss = nn.NLLLoss()#ignore_index=train_set.struct_action_vocab.stoi[train_set.pad])
        lex_loss           = nn.NLLLoss()#ignore_index=train_set.lex_vocab.stoi[train_set.pad])
        struct_loss        = nn.NLLLoss()#ignore_index=train_set.struct_vocab.stoi[train_set.pad])
        #reduction='mean'
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        scheduler = LambdaLR(optimizer,lr_lambda = lambda epoch:learning_rate/(1+epoch))

        for e in range(epochs): 

            scheduler.step()
            
            _lex_loss,_lex_action_loss, _struct_action_loss, _struct_loss = 0,0,0,0
            N = 0
            print('train')
            dataloader = BucketLoader(train_set,batch_size,device,alpha)
            for batch in dataloader:
                print('train xtokens',batch.xtokens)

                self.zero_grad()

                seq_representation =  self.forward_base(batch.xtokens,batch.tokens_length)

                
                pred_lexaction     =  self.forward_lexical_actions(seq_representation)
                pred_structaction  =  self.forward_structural_actions(seq_representation)
                pred_ytokens       =  self.forward_lexical_tokens(seq_representation)
                pred_structlabels  =  self.forward_structural_labels(seq_representation)

                print('train slabels',pred_structlabels)
                print('train reflabels',ref_structlabels)
                
                ref_lexactions     =  batch.lex_actions.view(-1)      #flattens the target too
                ref_structactions  =  batch.struct_actions.view(-1)   #flattens the target too
                ref_ytokens        =  batch.ytokens.view(-1)          #flattens the target too
                ref_structlabels   =  batch.struct_labels.view(-1)    #flattens the target too
                
                loss1 = lex_action_loss(pred_lexaction,ref_lexactions)       
                loss2 = struct_action_loss(pred_structaction,ref_structactions)       
                loss3 = lex_loss(pred_ytokens,ref_ytokens)       
                loss4 = struct_loss(pred_structlabels,ref_structlabels)       

                _lex_loss           += loss3.item()
                _lex_action_loss    += loss1.item()
                _struct_loss        += loss4.item()
                _struct_action_loss += loss2.item()
                N += sum(batch.tokens_length)
                
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                loss3.backward(retain_graph=True)
                loss4.backward()
                optimizer.step()

            L = _lex_loss + _lex_action_loss + _struct_action_loss + _struct_loss
            print("Epoch",e,'training loss (NLL) =', L/(4*N),'learning rate =',scheduler.get_lr()[0],N)
            print('        lex loss           (NLL) = ',_lex_loss/N)
            print('        lex action loss    (NLL) = ',_lex_action_loss/N)
            print('        struct loss        (NLL) = ', _struct_loss/N)
            print('        struct action loss (NLL) =',_struct_action_loss/N)
            #Development f-score computation
            #pred_trees = list(tree for (derivation,tree) in self.predict(dev_set,batch_size))
            pred_trees = list(tree for (derivation,tree) in self.predict(dev_set,batch_size,device))
            #for t in pred_trees[:10]:
            #    print(t)
            fscores    = [ reftree.compare(predtree)[2]   for (predtree,reftree) in zip(pred_trees,dev_set.tree_set) ]
            print("        development F-score = ", sum(fscores) / len(fscores))
            
    @staticmethod 
    def derivation2tree(derivation):
        """
        This builds up a ConsTree from a derivation
        Args: 
           derivation (list):a list of actions
        Returns:
           ConsTree. a constituent tree object built from the derivation
        """
        derivation.reverse()
        tree_stack = [ ]
        while derivation:
            action,label = derivation.pop()
            if action == LCmodel.ACTION_SHIFT_INIT :
                tree_stack.append( ConsTree(label) )
            elif action == LCmodel.ACTION_PREDICT:
                left_child = tree_stack.pop()
                tree = ConsTree(label,children=[ left_child ])  
                tree_stack.append(tree)
            elif action == LCmodel.ACTION_SHIFT_ATTACH:
                LCmodel.rightmostNT(tree_stack[-1]).add_child(ConsTree(label))
            elif action == LCmodel.ACTION_ATTACH:
                child = tree_stack.pop()
                child = ConsTree(label,children=[child])
                LCmodel.rightmostNT(tree_stack[-1]).add_child(child)
            else:
                print('derivation error')
        return tree_stack[-1] 
                
    @staticmethod
    def oracle_derivation(ctree,left_corner=True):
        """ 
        This computes an oracle derivation from a constituent tree.
        The tree must be strictly binarized. (Only binary rules and
        nothing else)
        Args: 
           ctree   (ConsTree): the tree to extract the derivation from
           left_corner (bool): the tree is a current left corner
           root        (bool): the tree is the topmost node in the overall tree
        Returns:
           a list of actions.  
        """
        if ctree.is_leaf( ): 
            return [(LCmodel.ACTION_SHIFT_INIT,ctree.label)] if left_corner else [(LCmodel.ACTION_SHIFT_ATTACH,ctree.label)]
        else: #internal node
            derivation = LCmodel.oracle_derivation(ctree.get_child(0),left_corner=True)
            if left_corner: 
                derivation.append((LCmodel.ACTION_PREDICT,ctree.label))
            else:
                derivation.append((LCmodel.ACTION_ATTACH,ctree.label))
            derivation.extend( LCmodel.oracle_derivation(ctree.get_child(1),left_corner=False) )
        return derivation 

def input_treebank(filename):
    """
    Data generator that reads and preprocesses a treebank from a file 
    Args:
       filename (string): a file where to read trees
    Yields:
       a sequence of trees 
    """
    istream = open(filename)
    for treeline in istream:
        tree = ConsTree.read_tree(treeline)
        tree.add_eos() 
        ConsTree.right_markovize(tree)
        ConsTree.close_unaries(tree)
        tree.strip_tags()
        yield tree
    istream.close()
    
def output_treebank(treelist,filename=None):
    """
    Prints a treelist to file 
    Args:
       treelist   (list): a list of trees
       filename (string): a file where to write trees
    """
    ostream = sys.stdout if filename is None else open(filename,'w')
    for tree in treelist:
        tree.expand_unaries()
        tree.unbinarize()
        tree.strip_eos()
        print(tree,file=ostream)
    if filename: 
        ostream.close()

        
if __name__ == '__main__':
    
    devset   =  [ '(TOP@S I (S: (VP love (NP em both)) .))']#,'(S (DP The (NP little monkey)) (VP screams loud))','(S (NP the dog) walks)','(S (NP a cat) (VP chases (NP the mouse)))','(S (NP A wolf) (VP eats (NP the pig)))']
    #print(treebank)
    trainset = list(input_treebank('../ptb_train.mrg'))
    #devset   = list(input_treebank('../ptb_dev.mrg'))

    #trainset   =  [ ConsTree.read_tree('(S (DP The (NP little monkey)) (VP screams loud))')]
    #devset     =  [ ConsTree.read_tree('(S (DP The (NP little monkey)) (VP screams loud))')]

    #train_df       = ParsingDataSet(trainset,min_lex_counts=10)
    #dev_df         = ParsingDataSet(devset,root_dataset=train_df)
    dev_df         = ParsingDataSet([ConsTree.read_tree(t) for t in devset])
    #print('Train Vocab size',train_df.lex_vocab.size())
    print('Dev   Vocab size',dev_df.lex_vocab.size())
    #print('Train label size',train_df.struct_vocab.size())
    #print('Train label size',train_df.struct_vocab.size(),train_df.struct_vocab.itos)
    print('Dev label size',dev_df.struct_vocab.size(),dev_df.struct_vocab.itos)
    
    parser = LCmodel(dev_df,rnn_memory=300,embedding_size=100,device=3)
    parser.cuda(device=3)
    parser.train(dev_df,dev_df,400,batch_size=1,learning_rate=5.0,device=3,alpha=0.0)  
 
