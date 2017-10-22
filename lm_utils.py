from math import ceil
from random import shuffle



class RNNLMGenerator:
    """
    This class wraps the coded data management for an NNLM (unstructured model)
    Each datum has the form X=(x1,x2,x3,...), Y=(x2,x3,x4,...) and is coded on integers.
    The generator shuffles the data sentence-wise but does not change intra-sentential word-order.
    """
    def __init__(self,X,Y,unk_word_code,batch_size=1):
        """
        @param X,Y the encoded X,Y values as lists (of lists for both X and Y)
        @param unk_word_code : the integer xcode for unknwown words
        @param batch_size:size of generated data batches (in tokens !)
        """
        assert(len(X)==len(Y))
        assert(all(len(x) == len(y) for x,y in zip(X,Y)))

        self.X,self.Y = X,Y                   # X,Y are couples of sentences
        self.idxes = list(range(len(self.X))) #indexes the sentences
        self.batch_size = batch_size          #number of tokens in a batch
        self.start_idx = 0                    #token indexing
        self.unk_x_code = unk_word_code
        self.randomize_corpus()

        
    def randomize_corpus(self):
        """
        Shuffles the corpus sentence-wise
        """
        shuffle(self.idxes)
        self.Xtoks = [ x for idx in self.idxes for x in self.X[idx] ]
        self.Ytoks = [ y for idx in self.idxes for y in self.Y[idx] ]
        self.Ntoks = len(self.Ytoks)

    def select_indexes(self,random_restart=True):
        """
        Internal function for selecting sentential indexes to process.
        @param random_restart: if all data is consumed, then reshuffle and restart.
        @return an interval of indexes to provide
        """
        end_idx = self.start_idx+self.batch_size
        
        if self.start_idx == self.Ntoks and random_restart:
            self.randomize_corpus()
            self.start_idx = 0
            end_idx = self.batch_size
            
        if end_idx > self.Ntoks:
            end_idx = self.Ntoks

        sidx = self.start_idx
        self.start_idx = end_idx
        return (sidx,end_idx)


    def get_num_batches(self):
        """
        returns the number of batches for this generator without random restart 
        """
        return ceil(self.Ntoks / self.batch_size)
    
    def has_next_batch(self):
        """
        True if some data can be provided by a call to next_batch
        without random restart 
        """
        return self.start_idx != self.Ntoks

    def batch_all(self):
        """
        Returns the whole data set in its current order
        """
        return (self.Xtoks,self.Ytoks)

    def next_batch(self,random_restart=True):
        """
        A data generator called by the fitting or predict functions.
        @param random_restart: if all data is consumed, then reshuffle and restart.
        @yield : an encoded subset of the data
        """
        while True:
            start_idx,end_idx = self.select_indexes(random_restart)
            X     = self.Xtoks[start_idx:end_idx]
            Y     = self.Ytoks[start_idx:end_idx]
            yield (X,Y)

    def get_num_sentences(self):
        return len(self.X)
    
    def next_sentence(self):
        """
        A data generator called by the fitting or predict functions
        @yield a sentence as a couple (X,Y) in the natural order of the corpus
        """
        idx = 0
        while True:
            if idx == self. get_num_sentences():
                idx = 0
            yield (self.X[idx],self.Y[idx])

            
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
        self.idxes = list(range(self.N)) 
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
            X     = self.Xtoks[start_idx:end_idx]
            Y     = self.Ytoks[start_idx:end_idx]
            yield (X,Y)


if __name__ == '__main__':
    #Exemple usage (basic autoencoder)
    #TODO
    pass
