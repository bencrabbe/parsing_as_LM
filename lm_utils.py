from math import ceil
from random import shuffle

class NNLMGenerator:
    """
    This class wraps the coded data management for an NNLM (unstructured model)
    Each datum has the form (x1,x2,x3),y and is coded on integers.
    """
    def __init__(self,X,Y,unk_word_code,batch_size=1):
        """
        @param X,Y the encoded X,Y values as lists (of lists for both X and Y)
        @param unk_word_code : the integer xcode for unknwown words
        @param batch_size:size of generated data batches
        """
        assert(len(X)==len(Y))
        self.X,self.Y = X,Y
        self.N = len(Y)
        self.idxes = list(range(self.N)) #indexes the sentences
        self.batch_size = batch_size     #number of tokens in a batch
        self.start_idx = 0               #token indexing
        self.unk_x_code = unk_word_code
  
    def select_indexes(self,random_restart=True):
        """
        Internal function for selecting indexes to process.
        @param random_restart: if all data is consumed, then reshuffle and restart.
        @return an interval of indexes to provide
        """
        end_idx = self.start_idx+self.batch_size
        
        if self.start_idx == self.N and random_restart:
            shuffle(self.idxes)
            self.start_idx = 0
            end_idx = self.batch_size
            
        if end_idx > self.N:
            end_idx = self.N

        sidx = self.start_idx
        self.start_idx = end_idx
        return (sidx,end_idx)


    def get_num_batches(self):
        """
        returns the number of batches for this generator without random restart 
        """
        return ceil(self.N / self.batch_size)
    
    def has_next_batch(self):
        """
        True if some data can be provided by a call to next_batch
        without random restart 
        """
        return self.start_idx != self.NN

    def batch_all(self):
        """
        Returns the whole data set in its natural order
        """
        return (self.X,self.Y)
        
            
    def next_batch(self,random_restart=True):
        """
        A generator called by the fitting function.
        @param random_restart: if all data is consumed, then reshuffle and restart.
        @yield : an encoded subset of the data
        """
        while True:
            start_idx,end_idx = self.select_indexes(random_restart)
            X     = self.X[start_idx:end_idx]
            Y     = self.Y[start_idx:end_idx]
            yield (X,Y)

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
        
            
            
if __name__ == '__main__':
    #Exemple usage (basic autoencoder)
    pass
