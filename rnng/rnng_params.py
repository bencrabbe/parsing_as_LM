import configparser

"""
That's a module for loading external param files 
"""

#These are namespaces for the hyperparams of the RNNG parser
class StructParams:

    STACK_EMB_SIZE      = 150 #INPUT to main RNN = either tree embedding or word+char embedding
    STACK_HIDDEN_SIZE   = 200 #= word embedding + char hidden size
    WORD_EMBEDDING_SIZE = 200
    CHAR_EMB_SIZE       = 50
    CHAR_HIDDEN_SIZE    = 50
    
class TrainingParams:
    LEX_MAX_SIZE      = 10000   #for full PTB training
    NUM_EPOCHS        = 20
    LEARNING_RATE     = 0.01
    DROPOUT           = 0.3


def read_config(filename=None):
    """
    Reads an external configuration file and fills in the global params
    """
    if filename:
        config = configparser.ConfigParser()
        config.read(filename)
        struct = config['structure']
        StructParams.STACK_EMB_SIZE      =  int(struct['stack_embedding_size']) if 'stack_embedding_size' in struct else StructParams.STACK_EMB_SIZE
        StructParams.STACK_HIDDEN_SIZE   =  int(struct['stack_hidden_size'])    if 'stack_hidden_size' in struct else StructParams.STACK_HIDDEN_SIZE
        StructParams.WORD_EMBEDDING_SIZE =  int(struct['word_embedding_size'])  if 'word_embedding_size' in struct else StructParams.WORD_EMBEDDING_SIZE
        StructParams.CHAR_EMB_SIZE       =  int(struct['char_embedding_size'])  if 'char_embedding_size' in struct else StructParams.CHAR_EMB_SIZE
        StructParams.CHAR_HIDDEN_SIZE    =  int(struct['char_hidden_size'])     if 'char_hidden_size' in struct else StructParams.CHAR_HIDDEN_SIZE
        
        learning = config['learning']
        TrainingParams.LEX_MAX_SIZE  = int(learning['lex_max_size'])    if 'lex_max_size' in learning else TrainingParams.LEX_MAX_SIZE
        TrainingParams.NUM_EPOCHS    = int(learning['num_epochs'])      if 'num_epochs' in learning else TrainingParams.NUM_EPOCHS
        TrainingParams.LEARNING_RATE = float(learning['learning_rate']) if 'learning_rate' in learning else TrainingParams.LEARNING_RATE
        TrainingParams.DROPOUT       = float(learning['dropout'])       if 'dropout' in learning else TrainingParams.DROPOUT
    
