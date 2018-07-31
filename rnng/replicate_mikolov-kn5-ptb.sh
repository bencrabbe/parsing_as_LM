

#SRILM replication of Mikolov-KN5 -> (PPL 142.5 where Mikolov reports 141.2) 
export PATH=$PATH:$PWD/srilm/bin/i686-m64

ORD=5

python lexicons.py ptb_train.raw ptb_train.raw > ptb_train.unk
python lexicons.py ptb_train.raw ptb_dev.raw > ptb_dev.unk
python lexicons.py ptb_train.raw ptb_test.raw > ptb_test.unk

ngram-count -order $ORD -text ptb_train.unk -lm templm -kndiscount -interpolate -unk -gt3min 1 -gt4min 1

ngram -ppl ptb_dev.unk -lm templm -order $ORD -unk
ngram -ppl ptb_test.unk -lm templm -order $ORD -unk

rm ptb_train.unk
rm ptb_dev.unk
rm ptb_test.unk
