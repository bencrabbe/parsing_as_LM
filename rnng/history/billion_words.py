#!/usr/bin/env python

import os
import random
import sys
from collections import Counter
"""
This is a module dedicated to preprocess and extract stats from the 1 billion word corpus.
"""

def process_files(root_path):
    """
    Generator function.
    Yields billion words train filenames given the root path
    Args:
       root_path (string): the dirpath to the root of the billion words
    corpus (the README is in this dirpath)
    """
    train_path = '/'.join( [ root_path,'training-monolingual.tokenized.shuffled']) 
    for rel_file in os.listdir(train_path):
        yield os.path.join(train_path,rel_file)

def sample_file(root_path):
    """
    Samples a file name from the 1 billion words filenames
    Args:
       root_path (string): the dirpath to the root of the billion words
    corpus
    Yields : an absolute filename (as a string)
    """ 
    file_list = list(process_files(root_path))
    yield random.choice(file_list)
    
def normalise_token(token):
    return token

def next_sentence(filename):
    """
    Generator function.
    Yields the next sentence from filename or returns None if file is
    empty
    Returns.
       a list of strings. The tokenized sentence.
    """
    istream = open(filename)
    for sentence in istream:
        yield [normalise_token(token) for token in sentence.split()]
    istream.close()
    return None

def extract_vocabulary(root_path):
    """
    Returns a Counter with token counts in the whole corpus
    """
    vocabulary = Counter()
    for filename in process_files(root_path):
        print('processing file %s'%(filename,),file=sys.stderr)
        for sentence in next_sentence(filename):
            vocabulary.update(sentence)
        return vocabulary

if __name__ == '__main__': 
    vocab = extract_vocabulary('/home/bcrabbe/parsing_as_LM/rnng/history/billion_words')
    print('vocab size',len(vocab))
    print('vocab size (>=5)',len([tok for tok, count in vocab.items() if int(count) >= 5]))
    print('vocab size (>=10)',len([tok for tok, count in vocab.items() if int(count) >= 10]))
    print('vocab size (>=20)',len([tok for tok, count in vocab.items() if int(count) >= 20]))
    print('vocab size (>=50)',len([tok for tok, count in vocab.items() if int(count) >= 50]))
    print('vocab size (>=100)',len([tok for tok, count in vocab.items() if int(count) >= 100]))
    
