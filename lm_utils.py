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


if __name__ == '__main__':
    #Exemple usage (basic autoencoder)
    #TODO
