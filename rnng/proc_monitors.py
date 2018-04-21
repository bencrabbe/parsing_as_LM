#!/usr/bin/env python


import numpy as np
import pandas as pd

"""
This module implements monitoring tools for tracking measures of
interest for cognitive modeling
"""

class AbstractTracker(object):
    """
    That's the empty tracker. It does nothing but it defines the
    minimal interface one wants to subclass for getting measures from
    a parser
    """
    def __init__(self,wordlist=None):
        """
        @param wordlist: the list of word entries known to the parser
        """
        pass

    def log_configuration(self,configuration):
        """
        This tracks all the stats of interest from a lexical configuration at time step t
        @param configuration: a configuration of the parser at time step t (after generating a word)
        """
        pass

    def next_word(self):
        """
        Moves to the next word
        """
        pass
        
    def next_sentence(self,tokens):
        """
        Moves to the next sentence
        @param tokens: the list of tokens from the sentence
        """
        pass
