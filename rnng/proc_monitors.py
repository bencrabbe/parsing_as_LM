#!/usr/bin/env python

import pandas as pd
from collections import OrderedDict

class RuntimeStats:
    """
    Runtime stats is a utility class for counting and aggregating temporal stats from
    the parser. It emulates some dictionary behaviour with limited aggregate computations on the last time step
    """
    
    def __init__(self,init_dict):
        
        self.stats    = OrderedDict(init_dict)
        self.push()

    def copy(self):
        
        return RuntimeStats(self.stats)

    #FUNCTIONS CONTROLLING THE TIME STEP
    def push(self):
        for key in self.stats:
            self.stats[key].append(0)
        
    def peek(self):
        
        K = self.stats.keys()
        V = [self.stats[k][-1] for k in K]
        return list(zip(K,V))
    

    def get_dataframe(self):
        #returns the full pandas data frame 
        return pd.DataFrame.from_dict(self.stats)
            
    #FUNCTIONS OPERATING ON AGGREGATIONS OF THE LAST TIME STEP
    
    def __getitem__(self,key):
        """
        Returns the key value on top (overloads [] syntax)
        """
        return self.stats[key][-1]
        
    def __setitem__(self,key,value):
        """
        Sets the key value on top (overloads [] syntax)
        """
        assert(key in self.stats)
        self.stats[key][-1] = value
            
    def __add__(self,other):
        """
        Returns a copy of this object added with another other
        RuntimeStats object (operates on top)
        """
        assert(len(set(self.stats.keys()) - set(other.stats.keys())) == 0)
        #both objects must have the exact same keys
        
        result = self.copy()                
        for key in other.stats.keys():
            result.stats[key] += other[key] 
        
    def __iadd__(self,other):
        """
        In-place addition with other object (operates on top)
        """
        assert(len(set(self.stats.keys()) - set(other.stats.keys())) == 0)

        for key in other.stats.keys():
            self.stats[key] += other[key] 

if __name__ == '__main__':

    s = RuntimeStats({'LL':0,})
    
