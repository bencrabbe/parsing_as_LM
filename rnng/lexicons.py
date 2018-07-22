#! /usr/bin/env python
from collections import Counter
import json
import numpy as np

"""
These are various lexer style utilities for coding strings to integers
and vice versa.
"""
class SymbolLexicon:
    """
    This class manages encoding of a lexicon as a set of finite size, including unknown words
    management. It provides services for mapping tokens to integer
    indexes and vice-versa.
    """
    def __init__(self,wordlist,unk_word=None,special_tokens=[ ],count_threshold=-1,max_lex_size=100000000):
        """
        @param wordlist       : a list of strings or a collections.Counter
        @param unk_word       : a token string for unknown words
        @param special tokens : a list of reserved tokens such as <start> or <end> symbols etc that are added to the lexicon 
        @param count_threshold: words are part of the lexicon if their counts is > threshold
        @param max_lex_size   : max number of elements in the lexicon
        """
        counts       = wordlist if isinstance(wordlist,Counter) else Counter(wordlist)
        lexlist      = [ word for word, c in counts.most_common(max_lex_size) if c > count_threshold ]

        if unk_word:
            lexlist.append(unk_word)
        self.UNK     = unk_word

        lexlist.extend(special_tokens)
        self.words2i = dict([ (w,idx) for idx,w in enumerate(lexlist)])
        self.i2words = lexlist

    def get_UNK_ID(self):
        """
        @return the integer index of the unknown word
        """
        if self.UNK:
            return self.words2i[self.UNK]
        else:
            return None
        
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
        return token if token in self.words2i else self.UNK

    def index(self,token):
        """
        @param token: a string
        @return the index of the word in this lexicon
        """
        return self.words2i[ self.normal_wordform(token) ]
        
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
        ostream.write(json.dumps({'UNK':self.UNK ,'lex':self.i2words}))
        ostream.close()

    @staticmethod
    def normalize_brown_file(cls_filename,lexical_set,out_filename,UNK_SYMBOL='<UNK>'):
        """
        This intersects the content of cls_filename with lexical_set
        and prints it in out_filename.
        It additionnaly adds an UNK word and an UNK cluster code
        @param cls_filename : a Brown cluster file
        @param lexical_set  : a set of strings
        @param outfilename  : the new Brown cluster file
        @param UNK_SYMBOL.  : the UNK word symbol 
        """
        istream = open(cls_filename)
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

        
    @staticmethod
    def unkify_brown_file(cls_filename,out_filename,UNK_SYMBOL='<UNK>'):
        istream = open(cls_filename)
        ostream = open(out_filename,'w')
        s = istream.read() + '0\t%s\t%d'%(UNK_SYMBOL,1)
        ostream.write(s)
        istream.close()
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
        lex.words2i = dict([ (w,idx)    for idx,w in enumerate(lexlist)])
        lex.i2words = lexlist
        lex.UNK     = UNK
        return lex

 

if __name__ == '__main__':
    symlex = SymbolLexicon(["A"]*3+['B']*4+['C'],count_threshold=1)
    print(symlex)
    print(symlex.index('A'),symlex.index('B'),symlex.index('C'),symlex.index('D'))
    print(symlex.wordform(0),symlex.wordform(1),symlex.wordform(2))
    print(blex)
