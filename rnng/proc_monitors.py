#!/usr/bin/env python

import pandas as pd

class RuntimeStats:
    """
    Runtime stats is a utility class for counting and aggregating temporal stats from
    the parser. It emulates some dictionary behaviour with limited aggregate computations on the last time step
    """
    
    def __init__(self,*args):

        self.stats    = {}
        self.okeys    = tuple([])
        if args:
            self.okeys = args
            for key in args:
                self.stats[key] = []

    def copy(self):
        
        r = RuntimeStats()
        r.stats = dict([(k,v[:]) for k,v in self.stats.items()])
        r.okeys = self.okeys[:]
        return r

    
    #FUNCTIONS CONTROLLING THE TIME STEP
    def push_row(self,**kwargs):
        """
        Adds/pushes a new row on top of the table.
        Defaults to adding a row of 0.0 but one may specify values as kwargs 
        """
        if kwargs:
            assert( len(set(self.okeys) - set(kwargs.keys())) == 0)
            for key,value in kwargs.items():
                self.stats[key].append(value)
        else:
            for key in self.stats:
                self.stats[key].append(0)
        
    def peek(self):
        """
        Returns a tuple of values from the last (top) row, key ordering is guaranteed
        """
        #guarantees values are ordered according to the kzys
        V = [self.stats[k][-1] for k in self.okeys]
        return tuple(V)
    

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

    def __radd__(self,other):
        return other.__add__(self)
    
    def __add__(self,other):
        """
        Returns a copy of this object added with another other
        RuntimeStats object (operates on top)
        """
        assert(len(set(self.stats.keys()) - set(other.stats.keys())) == 0)
        #both objects must have the exact same keys
        
        result = self.copy()                
        for key in other.stats.keys():
            result.stats[key][-1] += other.stats[key][-1] 
        return result
    
    def __iadd__(self,other):
        """
        In-place addition with other object (operates on top)
        """
        assert(len(set(self.stats.keys()) - set(other.stats.keys())) == 0)

        for key in other.stats.keys():
            self.stats[key][-1] += other.stats[key][-1] 
        return self

    
if __name__ == '__main__':

    s = RuntimeStats('LL','acc')
    t = RuntimeStats('LL','acc')
    s.push_row(LL=0,acc=1)
    t.push_row(LL=1,acc=1)
    print(t['acc'],s['LL'])
    y = t+s
    print(t.peek())
    print(s.peek(),y.peek())
    #s += (y+RuntimeStats(LL=1,acc=1))
    #print(s.peek())
