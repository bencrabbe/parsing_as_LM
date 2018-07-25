#!/usr/bin/env python


import numpy as np
import pandas as pd

class RuntimeStats:
    """
    Runtime stats is a utility class for collecting and gathering stats from
    the parser. It emulates some dictionary behaviour with limited computations 
    """
    
    def __init__(self):
        self.stats = {}

    def __getitem__(self,key):
        pass
    
    def __setitem__(self,key,value):
        pass

    def __add__(self,other):
        pass
    
    def __iadd__(self,other):
        pass

    def get_data(self):
        #returns the full data frame as pandas or csv
        pass

    def last_data(self):
        pass
