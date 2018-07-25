#!/usr/bin/env python


import numpy as np
import pandas as pd
from collections import OrderedDict

class RuntimeStats:
    """
    Runtime stats is a utility class for counting, collecting and gathering stats from
    the parser. It emulates some dictionary behaviour with limited computations 
    """
    
    def __init__(self,init_dict={}):
        
        self.stats = OrderedDict(init_dict)

    def copy(self):
        
        return RuntimeStats(self.stats)
                
    def __getitem__(self,key):
        """
        Returns the key value (overloads [] syntax)
        """
        return self.stats[key]
        
    def __setitem__(self,key,value):
        """
        Sets the key value (overloads [] syntax)
        """
        return self.stats[key] = value

    def __add__(self,other):
        """
        Returns a copy of this object added with another other RuntimeStats object
        """
        result = self.copy()
        
        for key in other.stats.keys():
            if key in result.stats:
                result.stats[key] += other[key] 
            else:
                self.stats[key]    = other[key] 

                
    def __iadd__(self,other):
        """
        In-place addition with other object
        """
        for key in other.stats.keys():
            if key in self.stats:
                self.stats[key] += other[key] 
            else:
                self.stats[key] = other[key] 

    def get_data(self):
        #returns the full data frame as pandas or csv
        pass

    def last_data(self):
        pass
