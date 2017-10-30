

#SRILM replication of Mikoklov-KN5 -> (PPL 142.5 where Mikolov reports 141.2) 
export PATH=$PATH:$PWD/srilm/bin/i686-m64

ORD=5

ngram-count -order $ORD -text $PWD/ptb/ptb_deps.train.txt -lm templm -kndiscount -interpolate -unk -gt3min 1 -gt4min 1

ngram -ppl $PWD/ptb/ptb_deps.dev.txt -lm templm -order $ORD -unk
ngram -ppl $PWD/ptb/ptb_deps.test.txt -lm templm -order $ORD -unk
