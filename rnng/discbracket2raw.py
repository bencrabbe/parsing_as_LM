from discotree import *

istream = open("negra/train.mrg")
ostream = open("negra/train.raw") 

for line in istream:

    t = DiscoTree.read_tree(line)
    print(' '.join(t.words(),file=ostream)

istream.close()
ostream.close()


