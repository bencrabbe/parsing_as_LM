import numpy as np

from collections import Counter
from random import shuffle

from keras.models import Sequential
from keras.engine import Layer,InputSpec
from keras.layers import Input,Embedding, Dense,Activation,Flatten
from keras.optimizers import SGD
from keras import backend as K


class NumericDataGenerator:
    """
    Stores an encoded treebank and provides it with a python generator interface
    """
    def __init__(self,X,Y,batch_size,nclasses,unk_word_code):
        """
        @param X,Y the encoded X,Y values as lists (of lists for X)
        @param batch_size:size of generated data batches
        @param nclasses: number of Y classes
        @param unk_word_code : the integer xcode for unknwown words
        """
        assert(len(X)==len(Y))
        self.X,self.Y = np.array(X),Y
        self.nclasses = nclasses
        self.N = len(Y)
        self.idxes = list(range(self.N))
        self.batch_size = batch_size
        self.start_idx = 0
        self.unk_x_code = unk_word_code
        self.x_counts(self.X)
        
    def x_counts(self,X):
        self.word_freqs = Counter() 
        for line in X:
            self.word_freqs.update(line)


    #you might want to avoid resampling reserved tokens such as START OR END SENTENCE markers
    def sample_one(self,x,alpha):
        """
        Sample x data for unk words
        """
        f = alpha / (self.word_freqs[x]+alpha)
        if rand() < f:
            return self.unk_x_code
        else:
            return x
        
    def sample_unknowns(self,X,alpha):

        return [ [self.sample_one(elt,alpha) for elt in line]  for line in X]
       

    def oov_rate(self,X=None):
        """
        Returns the oov rate of this data set
        """
        if X == None:
            X = self.X
        Nx =  len(X)*len(X[0])
        return sum( sum(x == self.unk_x_code for x in line ) for line in X) / Nx
    
    def guess_alpha(self,oov_rate,epsilon=0.01):
        """
        Provides a guess of the word drop alpha value from a dataset
        and an oov rate
        Uses the oov rate to compute alpha estimation from the dataset
        @epsilon: the precision of the estimation for the alpha value
        @return alpha (0 <= alpha <= infty)
        """
        print('guessing alpha lex...')
        Nx =  len(self.X)*len(self.X[0])
        if oov_rate == 0 :
            return 0
            
        amin,amax = (0.0, 20.0)
        
        while (amax - amin) > epsilon:
            alpha = (amax+amin) / 2
            print(alpha)
            genX = self.sample_unknowns(self.X,alpha)
            gen_oov = self.oov_rate(X=genX)
            if gen_oov > oov_rate:
                amax = alpha
            elif gen_oov < oov_rate:
                amin = alpha
        return alpha
                
                
    
    def select_indexes(self,randomize=True):
        """
        Internal function for selecting samples indexes.
        @param: if randomize is True, shuffles the
        dataset when it is fully processed and restart processing at
        the beginning. if randomize is False, there is no restart.
        Returns a couple where start_idx == end_idx to indicate termination.
        """
        end_idx = self.start_idx+self.batch_size
        if end_idx > self.N:
            if randomize:
                shuffle(self.idxes)
                self.start_idx = 0
                end_idx = self.batch_size
            else:
                end_idx = self.N
                
        sidx = self.start_idx
        self.start_idx = end_idx
        return (sidx,end_idx)

    def generate(self,alpha=0.0,test=False):
        """
        A generator called by the fitting function.
        @alpha: param ( >= 0 ) indicating the rate of sampling for OOV.
        @test : generate the data sequentially by batches without
        restarting when the data set is fully consumed alpha is
        automatically set to 0.0. if test is False, it
        is a perpetual generator.
        @yield : an encoded subset of the data or False if end of data
        is reached 
        """
        if test:
            while True:
                start_idx,end_idx = self.select_indexes(randomize=False)
                if start_idx==end_idx:
                    yield False
                Y = np.zeros((self.batch_size,self.nclasses))
                X     = self.X[start_idx:end_idx]
                yvals = self.Y[start_idx:end_idx]
                for i,y in enumerate(yvals):
                    Y[i,y] = 1.0
                yield (X,Y)
        else:
            while True:
                start_idx,end_idx = self.select_indexes()
                Y = np.zeros((self.batch_size,self.nclasses))
                X     = self.X[start_idx:end_idx]
                yvals = self.Y[start_idx:end_idx]

                if alpha > 0.0:#dynamic resampling for unk words
                    yvals = [   self.sample_one(elt,alpha) for elt in yvals ]
                    for idx in range(len(X)):
                        for jdx in range(len(X[idx])):
                            X[idx,jdx] = self.sample_one(X[idx,jdx],alpha) 
                        
                for i,y in enumerate(yvals):
                    Y[i,y] = 1.0
                yield (X,Y)



class EmbeddingTranspose(Layer):
    """
    This is a dense layer for outputting word predictions designed to be used
    with weights tied to an embedding layer.
    """
    def __init__(self,output_dim,tied_layer,**kwargs):
        super(EmbeddingTranspose,self).__init__(**kwargs)
        self.output_dim  = output_dim   #vocabulary size
        self.tied        = tied_layer

    def build(self,input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        assert(input_dim == self.tied.output_dim) #!!! the geometry of this mirror layer must match the (mirror) geometry of the original layer
        assert(self.output_dim == self.tied.input_dim) #!!! the geometry of this mirror layer must match the (mirror) geometry of the original layer
        self.kernel = self.tied.embeddings
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)
        
    def call(self,x):
        output =  K.dot(x,K.transpose(self.kernel))
        return output


if __name__ == '__main__':
    #Exemple usage (basic autoencoder)
    
    #tied layers
    inputL  = Embedding(8,4,input_length=2)
    outputL = EmbeddingTranspose(8,inputL)

    #model
    model = Sequential()
    model.add(inputL)
    model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('tanh'))
    model.add(outputL)
    model.add(Activation('softmax'))

    sgd = SGD(lr=1)
    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())

    #fitting 
    x = np.array([ [idx,idx]  for idx in range(8)])
    y = np.eye(8)

    model.fit(x,y,epochs=500)
    print(model.predict(x))
