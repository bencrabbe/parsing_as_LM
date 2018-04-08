

#These are namespaces for the hyperparams of the RNNG parser
class StructParams:

    OUTER_HIDDEN_SIZE  = 180
    STACK_EMB_SIZE     = 256
    STACK_HIDDEN_SIZE  = 256 
    
    
class TrainingParams:
    LEX_MAX_SIZE      = 10000
    NUM_EPOCHS        = 1
    LEARNING_RATE     = 0.001
    DROPOUT           = 0.3
