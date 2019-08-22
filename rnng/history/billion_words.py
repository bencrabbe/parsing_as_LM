#!/usr/bin/env python

import os

"""
This is a module dedicated to preprocess and extract stats from the 1 billion word corpus.
"""

def process_files(root_path):
    """
    Yields billion words train filenames given the root path
    Args:
       root_path (string): the dirpath to the root of the billion words
    corpus (the README is in this dirpath)
    """
    train_path = '/'.join([ root_path,'training-monolingual.tokenized.shuffled']) 
    for rel_file in os.listdir(train_path):
        yield os.path.join(train_path,rel_file)
        

print(list(process_files('/home/bcrabbe/parsing_as_LM/rnng/history/billion_words')))
