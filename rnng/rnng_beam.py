class BeamElement:

    """
    This class is a place holder for elements in the beam.
    """
    def __init__(self,prev_element,prev_action,prefix_gprob,prefix_dprob):
        """
        Args:
             prev_element (BeamElement) : the previous element or None
             prev_action       (string) : the action generating this element or None
             prefix_gprob       (float) : prefix generative probability
             prefix_dprob       (float) : prefix discriminative probability
        """
        self.prev_element = prev_element
        self.prev_action  = prev_action
        self.prefix_gprob = prefix_gprob
        self.prefix_dprob = prefix_dprob
        self.configuration = None
        
    @staticmethod
    def init_element(configuration):
        """
        Args:
           configuration (tuple): the parser init config
        Returns:
           BeamElement to be used at init
        """
        b = BeamElement(None,None,0,0)
        b.configuration = configuration
        
    def is_initial_element(self):
        """
        Returns:
            bool. True if the element is root of the beam
        """
        return self.prev_element is None or self.prev_action is None
    
    def exec_action(self,parser,sentence):
        #MOVE AWAY : problematic for many reasons
        """
        Generates the element's configuration and assigns it internally.

        Args:
             sentence (list): a list of strings, the tokens.
             parser 
        Returns:
           A configuration
        """
        configuration = self.prev_element.configuration
        S,B,n,stack_state,lab_state = configuration
        
        if lab_state == RNNGparser.WORD_LABEL:
            self.configuration = self.generate_word(configuration,sentence)
        elif lab_state == RNNGparser.NT_LABEL:
            self.configuration = self.label_nonterminal(configuration,self.prev_action)
        elif self.prev_action == RNNGparser.CLOSE:
            self.configuration = self.close_action(configuration)
        elif self.prev_action == RNNGparser.OPEN:
            self.configuration = self.open_action(configuration)
        elif self.prev_action == RNNGparser.SHIFT:
            self.configuration = self.shift_action(configuration)
        elif self.prev_action == RNNGparser.TERMINATE:
            self.configuration = configuration       


            
class RNNGbeam:

    """
    Data Structure supporting beam search and methods for extracting information out of the beam.
    Allows delayed lazy execution of actions.
    """
    def __init__(self,init_element):
        """
        Args:
            init_element (BeamElement): the init element of the beam
        """
        self.beam = [[init_element]]
        self.successes = []


    def has_top(self):
        """
        Returns a bool indicating if the top of the beam is empty
        """
        return len(self.beam[-1]) > 0

        
    def enumerate_top(self):
        """
        Enumerates the elements from the top beam
        
        Yields:
           BeamElement
        """
        for elt in self.beam[-1]:
            yield elt

    def push_top(self,elt_list):
        """
        Pushes a new empty beam on top for next time step.
        Args:
            elt_list (list): list of BeamElement
        """
        self.beam.append(elt_list)
        
    def truncate(self,K):
        """
        Keeps only the K top elements of the top beam.
        Inplace destructive operation.

        Args:
             K (int): the number of elts to keep in the Beam
        """
        self.beam[-1] = self.beam[-1][:K]

    def discriminative_sample(self,K):
        """
        Samples without replacement K elements in the beam proportional to their discriminative probability
        Inplace destructive operation.
        Args:
             K (int): the number of elts to keep in the Beam
        """
        probs = np.exp(np.array([elt.prefix_dprob  for elt in self.beam[-1]]))
        samp_idxes = npr.choice(list(range(len(self.beam[-1]))),size=min(len(self.beam[-1]),K),p=probs,replace=False)
        self.beam[-1] = [ self.beam[-1][idx] for idx in samp_idxes]
        
    def generative_sort(self):
        """
        Sorts the beam elements by decreasing generative log prob.
        Inplace destructive operation.
        """
        self.beam[-1].sort(key=lambda x:x.prefix_gprob,reverse=True)

    def discriminative_sort(self):
        """
        Sorts the beam elements by decreasing discriminative log prob
        Inplace destructive operation.
        """
        self.beam[-1].sort(key=lambda x:x.prefix_dprob,reverse=True)

