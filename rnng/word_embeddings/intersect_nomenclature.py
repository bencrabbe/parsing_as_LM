#! /usr/bin/env python

import sys
import numpy as np

"""
This module intersects the nomenclature of a word embedding dictionary
with an existing nomenclature, hence generating a lightweight dictionary.
"""
def ptb_nomenclature(infilename):
    """
    Reads a ptb raw file and returns the nomenclature as a list of
    strings
    @param infilename: the ptb raw file
    @return a set of words 
    """
    istream = open(infilename)
    V = set([])
    for line in istream:
        V.update(line.split())
    istream.close()
    return V

def filter_w2v_nomenclature(w2vfilename,filter_nomenclature):
    """
    Reads a word2vec text file and returns the list of words that are
    in the filter_nomenclature
    @param w2filename: the name of a w2v text file
    @param filter_nomenclature: a set of strings
    @return a couple (wordlist,list of word vectors<-> a numpy matrix)
    """
    V = set(filter_nomenclature)
    wordlist = []
    veclist  = [] 
    istream = open(w2vfilename)
    istream.readline()          #skips w2v header
    for line in istream:
        fields = line.split()
        word = fields[0]
        if word in V:
            vec  = np.array([float(elt) for elt in fields[1:]])
            wordlist.append(word)
            veclist.append(vec)
    istream.close()
    M = np.array(veclist)
    return (wordlist,M)

def dump_w2vdictionary(outfilename,wordlist,matrix):
    """
    Dumps a w2v dictionary to a text file, using Mikolov text format
    @param outfilename: a string
    @param wordlist: a list of strings
    @param matrix : a matrix of vecs with as many row as words in the wordlist
    """
    ostream = open(outfilename,'w')
    print('%d %d'%(len(wordlist),matrix.shape[1]),file=ostream)
    for word,vec in zip(wordlist,matrix):
        print(' '.join([word]+ [ str(elt) for elt in vec]),file=ostream)
    ostream.close()
    
if __name__ == '__main__':
    
    import getopt

    nomfile = ''
    ifile   = ''
    ofile   = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"n:")
        ifile,ofile = args
    except:
        print ('intersect_nomenclature.py -n <raw_ptb_file> <w2v_inputfile> <w2v_outputfile>')
        sys.exit(1)
    
    for o,a in opts:
        if o == '-n':
            nomfile = a
        
    if nomfile and ifile and ofile:
        F =  ptb_nomenclature(nomfile)
        w,M = filter_w2v_nomenclature(ifile,F)
        dump_w2vdictionary(ofile,w,M)        
    else:
        print ('intersect_nomenclature.py -n <raw_ptb_file> <w2v_inputfile> <w2v_outputfile>')
