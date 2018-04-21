

#These are namespaces for the hyperparams of the RNNG parser
class StructParams:

    STACK_EMB_SIZE     = 300 
    STACK_HIDDEN_SIZE  = 256
    
class TrainingParams:
    LEX_MAX_SIZE      = 10000   #for full PTB training
    NUM_EPOCHS        = 2
    LEARNING_RATE     = 0.001
    DROPOUT           = 0.3
