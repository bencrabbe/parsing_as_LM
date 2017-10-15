

#SRILM replication of Mikoklov-KN5 -> (PPL 142.5 where Mikolov reports 141.2) 
export PATH=$PATH:$PWD/srilm/bin/i686-m64

ngram-count -order 5 -text $PWD/ptb/ptb_train_50w.txt -lm templm -kndiscount -interpolate -unk -gt3min 1 -gt4min 1
ngram -ppl $PWD/ptb/ptb_test.txt -lm templm -order 5 -unk
