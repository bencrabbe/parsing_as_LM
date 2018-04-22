#!/bin/sh

MKL_NUM_THREADS=4
NUM_EPOCHS=15

make_config(){
    #$1 NAME $2 = embedding size , $3 = lstm memory size $4 = dropout
    CNAME="$1.prm"
    echo > $CNAME 
    echo "[structure]" >> $CNAME
    echo "stack_embedding_size = $2" >> $CNAME
    echo "stack_hidden_size    = $3" >> $CNAME
    echo >> $CNAME
    echo "[learning]"  >> $CNAME
    echo "lex_max_size  = 10000"  >> $CNAME
    echo "num_epochs    = $NUM_EPOCHS" >> $CNAME
    echo "learning_rate = 0.0001" >> $CNAME
    echo "dropout       = $4"     >> $CNAME
}

train_brown(){
   # $1 = embedding size , $2 = lstm memory size $3 = dropout
   NAME="brown-$1-$2-$3-1000"  #1000 stands for the number of clusters
   mkdir -p $NAME
   make_config "$NAME/$NAME" $1 $2 $3
   #source activate py36
   #nohup python rnng.py -m $NAME -t ptb_train_mrg -d -ptb_dev.mrg -b ptb-1000.brown -c "$NAME.prm" >> $nohup.NAME.out &
   #mv $NAME.* $NAME/ 
}

#train_lexicalized(){

#}

#Brown clusters
train_brown 300 100 0.5
train_brown 300 200 0.5
train_brown 300 300 0.5

#Fully lexicalized
