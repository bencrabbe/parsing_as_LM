#!/bin/sh

MKL_NUM_THREADS=10
NUM_EPOCHS=20

make_config(){
    #$1 NAME $2 = stack embedding size , $3 = lstm memory size $4 = word embedding size $5 = dropout
    CNAME="$1.prm"
    echo > $CNAME  
    echo "[structure]" >> $CNAME
    echo "stack_embedding_size = $2" >> $CNAME
    echo "stack_hidden_size    = $3" >> $CNAME
    echo "word_embedding_size" = $4  >> $CNAME
    echo "char_embedding_size" = 50  >> $CNAME
    echo "char_hidden_size"    = 50  >> $CNAME

    echo >> $CNAME
    echo "[learning]"                  >> $CNAME
    echo "lex_max_size  = 10000"       >> $CNAME
    echo "num_epochs    = $NUM_EPOCHS" >> $CNAME
    echo "learning_rate = 0.1"         >> $CNAME
    echo "dropout       = $5"          >> $CNAME
}

make_lmconfig(){
    # $1 NAME $2 = embedding size , $3 = lstm memory size $4 = dropout
    CNAME="$1.prm"
    echo > $CNAME
    echo "[structure]"         >> $CNAME
    echo "embedding_size = $2" >> $CNAME
    echo "memory_size    = $3" >> $CNAME
    echo "[learning]"          >> $CNAME
    echo "dropout        = $4" >> $CNAME
    echo "learning_rate  = 0.1">> $CNAME
    echo "num_epochs     = $NUM_EPOCHS" >> $CNAME
}


train_rnng(){
   # $1 = stack embedding size , $2 = lstm memory size $3 = word embedding size $4 = dropout
   NAME="brown-$1-$2-$3-250"  #250 stands for the number of clusters
   mkdir -p $NAME
   make_config "$NAME/$NAME" $1 $2 $3 $4 
   source activate py36
   nohup python rnng.py -m $NAME/$NAME -t ptb_train.mrg -d -ptb_dev.mrg -b ptb-250.brown -c "$NAME/$NAME.prm" > "nohup.$NAME.out" & 
}

train_rnnlm(){
    # $1 = embedding size , $2 = lstm memory size $3 = dropout
    NAME="rnnlm-$1-$2-250"  #250 stands for the number of clusters
    mkdir -p $NAME
    make_lmconfig "$NAME/$NAME" $1 $2 $3 
    source activate py36
    nohup python rnnglmf.py -m $NAME/$NAME -t ptb_train.raw -d ptb_dev.raw -b ptb-250.brown -c "$NAME/$NAME.prm" > "nohup.$NAME.out"    
} 
train_rnnlm 300 250 0.5

#train_rnng 150 200 100 0.3
#train_rnng 250 300 200 0.3
#train_rnng  150 250 100 0.3
