#! /usr/bin/env python

from collections import Counter
import json
import sys
import numpy as np

"""
These are various lexer style utilities for coding strings to integers
and vice versa.
"""
class SymbolLexicon:
    """
    This class manages encoding of a lexicon as a set of finite size, including unknown words
    management. This behaves like a defaultdictionary. It provides services for mapping tokens to integer indexes and vice-versa.
    
    """
    def __init__(self,wordlist,unk_word=None):
        """
        @param wordlist       : a list of strings or a collections.Counter
        @param unk_word       : a token string for unknown words
        """
        if unk_word:
            wordlist.append(unk_word)

        #wordlist = list(set(wordlist))
        self.words2i  = dict([ (w,idx) for idx,w in enumerate(wordlist)])
        self.i2words  = wordlist
        self.unk_word = unk_word


        
    @staticmethod
    def make_lexicon_from_list(wordlist,unk_word=None,special_tokens=[ ],count_threshold=-1,max_lex_size=100000000):
        """
        Generates a lexicon from a corpus,only words with freq > count threshold are known to the dictionary.
        @param wordlist       : a list of strings or a collections.Counter
        @param unk_word       : a token string for unknown words
        @param special tokens : a list of reserved tokens such as <start> or <end> symbols etc that are added to the lexicon 
        @param count_threshold: words are part of the lexicon if their counts is > threshold
        @param max_lex_size   : max number of elements in the lexicon
        @return a SymbolLexicon
        """
        counts       = wordlist if isinstance(wordlist,Counter) else Counter(wordlist)
        lexlist      = [ word for word, c in counts.most_common(max_lex_size) if c > count_threshold ]

        lexlist.extend(special_tokens)
        return SymbolLexicon(lexlist,unk_word)
       
    def __str__(self):
        return ' '.join(self.words2i.keys())

    def __contains__(self,item):
        """
        Overrides the in operator
        @param item a word we want to know if it is known to the
        lexicon or not
        @return a boolean
        """
        return item in self.words2i
            
    def size(self):
        """
        @return the vocabulary size
        """
        return len(self.i2words)
        
    def normal_wordform(self,token):
        """
        @param token: a string
        @return a string, the token or the UNK word code
        """
        return token if token in self.words2i else self.unk_word

    def index(self,token):
        """
        @param token: a string
        @return the index of the word in this lexicon
        """
        return self.words2i[ self.normal_wordform(token) ]

    def unk_index(self):
        """
        @return the integer index of the unknown word if it exists
        """
        if self.unk_word:
            return self.index(self.unk_word)
        return -1
       
    def wordform(self,idx):
        """
        @param idx: the index of a token
        @return a string, the token for this index
        """
        return self.i2words[ idx ]
    
    def save(self,modelname):
        """
        @param modelname : the name of the model to save
        """
        ostream = open(modelname+'.json','w')
        ostream.write(json.dumps({'UNK':self.unk_word ,'lex':self.i2words}))
        ostream.close()

    @staticmethod
    def load(modelname):
        """
        @param modelname : the name of the model to load
        @return a SymbolLexicon object
        """
        istream = open(modelname+'.json')
        struct = json.loads(istream.read())
        istream.close()
        
        UNK     = struct['UNK']
        lexlist = struct['lex']
        
        lex = SymbolLexicon([])
        lex.words2i  = dict([ (w,idx)    for idx,w in enumerate(lexlist)])
        lex.i2words  = lexlist
        lex.unk_word = UNK
        return lex
    
def normalize_brown_file(brown_filename,lexical_set,out_filename,UNK_SYMBOL='<UNK>'):
    """
    This intersects the content of cls_filename with lexical_set
    and prints it in out_filename.
    It additionnaly adds an UNK word and an UNK cluster code
    @param brown_filename : a Brown cluster file
    @param lexical_set  : a set of strings
    @param outfilename  : the new Brown cluster file
    @param UNK_SYMBOL.  : the UNK word symbol 
    """
    istream = open(brown_filename)
    ostream = open(out_filename,'w')
    for line in istream:
        clsID,word,freq = line.split()
        if word in lexical_set:
            print('\t'.join([clsID,word,freq]),file = ostream)

    #THE UNK CLUSTER GETS ID 0
    uclsID = 0
    s = istream.read() + '%d\t%s\t%d'%(uclsID,UNK_SYMBOL,1)
    ostream.write(s)
    istream.close()
    ostream.close()
    return out_filename


def get_known_vocabulary(text,vocab_treshold=1):
    """
    This is a method common to all models in the package for acquiring the known lexicon.
    Args:
          text            (list): a list of strings (lines of the text or lists of tokens) or an open input stream 
          vocab_treshold   (int): words with counts > threshold are known to the vocab, otherwise unknown
    Returns:
          A set of string. The vocabulary 
    """
    lexicon = Counter()
    for line in text:
        if type(line) == str:
            lexicon.update(line.split())
        else:
            lexicon.update(line)
            
    return set([word for word, counts in lexicon.items() if counts > vocab_threshold])
        
 

if __name__ == '__main__':
    #Simulates the lookup procedure for all models in the package and prints back the result.
    #first arg = text where to acquire the vocab, following args = texts where to perform the simulation 

    istream = open(sys.argv[1])
    V = get_known_vocabulary(istream)
    istream.close()
    Lex = SymbolLexicon(list(V),unk_word='<UNK>')

    for text in sys.argv[2:]:
        istream = open(text)
        for line in istream:
           print(' '.join([Lex.normal_wordform(token) for token in line.split()]))
        istream.close()
    

    
