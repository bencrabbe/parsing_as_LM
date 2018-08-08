#!/bin/sh

MKL_NUM_THREADS=6
NUM_EPOCHS=20

make_config(){
    
    #$1 NAME $2 = stack embedding size , $3 = lstm memory size $4 = word embedding size $5 = dropout
    CNAME="$1.conf"
    echo > $CNAME  
    echo "[structure]" >> $CNAME
    echo "stack_embedding_size = $2" >> $CNAME
    echo "stack_hidden_size    = $3" >> $CNAME
    echo "word_embedding_size" = $4  >> $CNAME
    echo "char_embedding_size" = 50  >> $CNAME
    echo "char_hidden_size"    = 50  >> $CNAME

    echo >> $CNAME
    echo "[learning]"                  >> $CNAME
    echo "num_epochs    = $NUM_EPOCHS" >> $CNAME
    echo "batch_size    = 32"          >> $CNAME
    echo "learning_rate = 0.5"         >> $CNAME
    echo "dropout       = $5"          >> $CNAME
}

make_lmconfig(){
    # $1 NAME $2 = embedding size , $3 = lstm memory size $4 = dropout
    CNAME="$1.conf"
    echo > $CNAME
    echo "[structure]"         >> $CNAME
    echo "embedding_size = $2" >> $CNAME
    echo "memory_size    = $3" >> $CNAME
    echo "[learning]"          >> $CNAME
    echo "dropout        = $4" >> $CNAME
    echo "learning_rate  = 0.3">> $CNAME
    echo "num_epochs     = 25" >> $CNAME
}

train_rnng(){
   # $1 = stack embedding size , $2 = lstm memory size $3 = word embedding size $4 = dropout
   NAME="rnng-fr-$1-$2-$3-200"  #250 stands for the number of clusters
   mkdir -p $NAME
   make_config "$NAME/$NAME" $1 $2 $3 $4 
   source activate py36
   nohup python rnngf.py -m $NAME/$NAME -t ftb_train.mrg -d ftb_dev.mrg -b ftb-200.brown -c "$NAME/$NAME.conf" -p ftb_test.mrg -s > "nohup.$NAME.out" & #; \
   #python rnngf.py -m $NAME/$NAME -p prince/prince.en.txt -s >>  "nohup.$NAME.out") &
}

train_rnnlm(){
    # $1 = embedding size , $2 = lstm memory size $3 = dropout
    NAME="rnnlm-fr-$1-$2-250"  #250 stands for the number of clusters
    mkdir -p $NAME
    make_lmconfig "$NAME/$NAME" $1 $2 $3 
    source activate py36
    nohup python rnnglmf.py -m $NAME/$NAME -t ftb_train.raw -d ftb_dev.raw -b ftb-200.brown -c "$NAME/$NAME.conf" -p ftb_test.raw -s > "nohup.$NAME.out" & #;\
    # python rnnglmf.py -m $NAME/$NAME -p prince/prince.en.txt -s >>  "nohup.$NAME.out") &
} 

train_rnnlm 250 300 0.5
train_rnnlm 250 200 0.5
train_rnnlm 250 150 0.5


train_rnng 350 250 300 0.3
train_rnng 250 300 200 0.3
train_rnng 250 200 200 0.3
