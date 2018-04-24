from math import ceil
from random import shuffle



class RNNLMGenerator: 
    """
    This class wraps the coded data management for an NNLM (unstructured model)
    Each datum has the form X=(x1,x2,x3,...), Y=(x2,x3,x4,...) and is coded on integers.
    The generator shuffles the data sentence-wise but does not change intra-sentential word-order.
    """
    def __init__(self,X,Y,eos_code,batch_size=64,batch_width=40):
        """
        @param X,Y the encoded X,Y values as lists (of lists for both X and Y)
        @param unk_word_code : the integer xcode for unknwown words
        @param eos_code : the integer xcode for end of sentence words (used for padding mini-batches)
        @param batch_size:size of generated data batches (in sentences)
        """
        assert(len(X)==len(Y))
        assert(all(len(x) == len(y) for x,y in zip(X,Y)))

        self.X_stable,self.Y_stable = X,Y                   # X,Y are couples of sentences
        self.eos_code               = eos_code
        self.batch_size             = batch_size
        self.batch_width            = batch_width
        self.make_exact_batches()
        self.make_batches(max_width=batch_width)

        for elt in self.X_stable:
            print(elt)

        print()
        
    def get_num_sentences(self):
        return len(self.X_stable)
        
    def make_exact_batches(self):
        """
        Makes variable size batches based on exact sentences.
        Leads to inefficient excecution and/or memory blowup,
        but useful for computing exact metrics.
        """
        idxes = range(len(self.X_stable))
        
        #Bucketing
        self.buckets = {}
        for idx in idxes:
            L = len(self.X_stable[idx])
            if L in self.buckets:
                self.buckets[L].append(idx)
            else:
                self.buckets[L] = [idx]

    def get_num_exact_batches(self):
        return len(self.buckets)
    
    def next_exact_batch(self):
        """
        A data generator called by the fitting or predict functions.
        @yield a sentence as a couple (X,Y) in the natural order of the corpus
        """
        while True:
            for key, values in self.buckets.items():
                X = [self.X_stable[idx] for idx in values]
                Y = [self.Y_stable[idx] for idx in values]
                yield (X,Y)

                 
    def make_batches(self,max_width=40):
        """
        Makes batches for truncated backprop through time (TBTT)
        @param max_width: max size of a sentence in a batch.
        """
        self.X = self.X_stable[:]
        self.Y = self.Y_stable[:]

        #Truncate (modifies the data)        
        X = []
        Y = []
        MinRestSize = 5              #avoids the generation of too narrow batches
        for x,y in zip(self.X,self.Y):
            rest_x,rest_y = x,y
            while len(rest_x) > max_width:
                if len(rest_x) - max_width > MinRestSize:
                    X.append(rest_x[:max_width])
                    Y.append(rest_y[:max_width])
                    rest_x,rest_y = rest_x[max_width:],rest_y[max_width:]
                else:
                    break
            X.append(rest_x)
            Y.append(rest_y)
        self.X,self.Y = X, Y
            
        #Shuffle
        self.idxes = list(range(len(self.X)))
        shuffle(self.idxes)
        
        #Bucketing
        buckets = {}
        for idx in self.idxes:
            L = len(self.X[idx])
            if L in buckets:
                buckets[L].append(idx)
            else:
                buckets[L] = [idx]
                
        #Batching
        self.batches  = []
        current_batch = []
        for sent_length in range(max_width):
            if sent_length in buckets:
                examples = buckets[sent_length]
                while len(examples)+len(current_batch) >= self.batch_size:
                    split_idx = self.batch_size-len(current_batch)
                    current_batch.extend(examples[:split_idx])
                    self.batches.append(current_batch)
                    current_batch = []
                    examples = examples[split_idx:]
                current_batch.extend(examples)
        if len(current_batch) > 0:
            self.batches.append(current_batch)
                
        #Padding batches with different sized sentences
        for batch in self.batches:
            maxL = max([len(self.X[idx]) for idx in batch])
            for sent_idx in batch:
                if len(self.X[sent_idx]) < maxL:
                    padding = [self.eos_code] * (maxL - len(self.X[sent_idx]))
                    self.X[sent_idx] = padding + self.X[sent_idx]
                    self.Y[sent_idx] = padding + self.Y[sent_idx]

                    
    def get_num_batches(self):
        """
        returns the number of batches for this generator without random restart 
        """
        return len(self.batches)
    
    def next_batch(self):
        """
        A data generator called by the fitting or predict functions.
        @param random_restart: if all data is consumed, then reshuffle and restart.
        @yield : a batch of constant size and contant width
        """
        batches_idxes = list(range(len(self.batches)))
        shuffle(batches_idxes)
        _idx = 0 
        while True:
            if _idx == len(self.batches):
                self.make_batches(self.batch_width)
                shuffle(batches_idxes)
                _idx = 0
                
            sent_idxes = self.batches[batches_idxes[_idx]]
            X = [self.X[sidx] for sidx in sent_idxes]
            Y = [self.Y[sidx] for sidx in sent_idxes]
            yield (X,Y)
            _idx += 1

            
