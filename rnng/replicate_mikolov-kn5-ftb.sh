

#SRILM replication of Mikolov-KN5 -> (PPL 142.5 where Mikolov reports 141.2) 
export PATH=$PATH:$PWD/srilm/bin/i686-m64

ORD=5

python lexicons.py ftb_train.raw ftb_train.raw > ftb_train.unk
python lexicons.py ftb_train.raw ftb_dev.raw > ftb_dev.unk
python lexicons.py ftb_train.raw ftb_test.raw > ftb_test.unk

ngram-count -order $ORD -text ftb_train.unk -lm templm -kndiscount -interpolate -unk -gt3min 1 -gt4min 1

ngram -ppl ftb_dev.unk -lm templm -order $ORD -unk
ngram -ppl ftb_test.unk -lm templm -order $ORD -unk

rm ftb_train.unk
rm ftb_dev.unk
rm ftb_test.unk
