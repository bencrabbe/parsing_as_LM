import numpy as np

from keras.models import Sequential
from keras.engine import Layer,InputSpec
from keras.layers import Input,Embedding, Dense,Activation,Flatten
from keras.optimizers import SGD
from keras import backend as K




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
