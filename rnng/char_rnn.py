

class CharRNNBuilder:

    """
    That's a basic char rnn, that can be embedded into a larger model.
    This is a bi-lstm kind of model made to be used as a sub-model similarly to other dynet builders.
    """
    def __init__(self,char_embedding_size,memory_size,charset,model):
        """
        Creates a CharRNN from scratch. Should be used as a external constructor.
        Args:
            char_embedding_size          (int): size of char embeddings
            memory_size                  (int): size of RNN memory
            charset                      (str): a list of chars understood to be the charset of this model
            model  (dynet.ParameterCollection): parameters from the caller model
        """
        self.char_embedding_size = char_embedding_size
        self.memory_size         = memory_size
        self.model               = model
        self.charset             = SymbolLexicon(list(charset))
        self.allocate_structure()

    @staticmethod    
    def load_charset(self,modelname):
        """
        Loads a symbol lexicon from file
        Args:
            modelname (string):
        Returns:
            SymbolLexicon
        """
        return SymbolLexicon.load(modelname+'.char.codes')

    def save_charset(self,modelname):
        """
        Saves a symbol lexicon to file
        Args:
            modelname (string):
        """
        self.charset.save(modelname+'.char.codes')
        
    def allocate_structure(self):
        """
        Creates and allocates the network structure
        """
        #char input
        self.E    = self.model.add_lookup_parameters((self.charset.size(),self.char_embedding_size))
        #char output
        self.O    = self.model.add_parameters((self.char_embedding_size,self.memory_size*2))  
        self.b    = self.model.add_parameters(self.char_embedding_size)  
        #RNN
        self.fwd_rnn = dy.LSTMBuilder(1,self.char_embedding_size,self.memory_size,self.model)  
        self.bwd_rnn = dy.LSTMBuilder(1,self.char_embedding_size,self.memory_size,self.model)  

    def __call__(self,token):
        """
        Lookup alias for functional style.
        Args:
             token  (list): a list of chars
        Returns:
             a dynet expression. The char embedding.
        """
        return self.lookup(token)
    
    def lookup(self,token):
        """
        Performs forward propagation from the token yielding a char embedding
        Args:
             token  (list): a list of chars
        Returns:
             a dynet expression. The char embedding.
        """
        token = list(token)
        char_embeddings = self.E[self.charset.index(c) for c in token if token in charset] #ignores unk chars

        fwd_state       = self.fwd_rnn.initial_state()
        fwd_states      = fwd_state.transduce(char_embeddings)
               
        bwd_state       = self.bwd_rnn.initial_state()
        bwd_states      = bwd_state.transduce(reversed(char_embeddings))
        
        hidden          = dy.concatenate([fwd_states[-1].output(),bwd_states[-1].output()])
        return dy.rectify(self.O * hidden + self.b)


               
