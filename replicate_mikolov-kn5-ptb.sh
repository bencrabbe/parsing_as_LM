

#SRILM replication of KN5
export PATH=$(PATH)/srilm

ngram-count -order 5 -text ptb/ptb.train_50w.txt -lm templm -kndiscount
-interpolate -unk -gt3min 1 -gt4min 1
ngram -ppl ptb/ptb_test.txt -lm templm -order 5 -unk
