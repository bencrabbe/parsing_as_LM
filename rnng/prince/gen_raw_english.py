from constree import *

istream = open('prince/little-prince-english-stanford.txt')

for line in istream:
    print(' '.join(ConsTree.read_tree(line).tokens()))

istream.close()



 
