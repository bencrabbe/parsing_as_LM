

#These are namespaces for the hyperparams of the RNNG parser
class StructParams:

    OUTER_HIDDEN_SIZE  = 256
    STACK_EMB_SIZE     = 300 
    STACK_HIDDEN_SIZE  = 256
    
class TrainingParams:
    LEX_MAX_SIZE      = 10000   #for full PTB training
    NUM_EPOCHS        = 3
    LEARNING_RATE     = 0.0001
    DROPOUT           = 0.3
