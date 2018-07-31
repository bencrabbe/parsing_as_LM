

#SRILM replication of Mikoklov-KN5 -> (PPL 142.5 where Mikolov reports 141.2) 
export PATH=$PATH:$PWD/srilm/bin/i686-m64

ORD=5

ngram-count -order $ORD -text ptb_train.raw -lm templm -kndiscount -interpolate -unk -gt3min 1 -gt4min 1


ngram -ppl ptb_dev.raw -lm templm -order $ORD -unk
ngram -ppl ptb_test.raw -lm templm -order $ORD -unk
