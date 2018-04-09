

#These are namespaces for the hyperparams of the RNNG parser
class StructParams:

    OUTER_HIDDEN_SIZE  = 256
    STACK_EMB_SIZE     = 100
    STACK_HIDDEN_SIZE  = 100 
    
    
class TrainingParams:
    LEX_MAX_SIZE      = 1000
    NUM_EPOCHS        = 200
    LEARNING_RATE     = 0.0001
    DROPOUT           = 0.0001
