#! /usr/bin/env python
from collections import Counter
import json

class BrownLexicon:
    """
    This class manages brown cluster as generated by P. Liang's
    clustering package. By hypothesis the set of wordforms is
    partitioned by the clusters : no wordform can belong to more than
    one cluster.
    """
    def __init__(self,w2cls,word_counts,freq_threshold=1):
        """
        @param w2cls: a dict wordform -> cluster IDs (IDs as integers)
        @param word_counts: a Counter wordform -> counts
        @param max_lexicon_size: max number of wordforms to include in the dictionary
        """
        self.word_counts = dict( [ (w,count) for (w,count) in word_counts.items() if count > freq_threshold])
        self.w2cls       = dict(  [ (w,C) for (w,C) in w2cls.items() if w in self.word_counts])

        #computes the counts of the clusters
        self.cls_counts   = {}          #raw counts of the clusters in the corpus
        for word,clust in self.w2cls.items():
            C = self.get_cls(word,defaultval=None)
            if C :
                self.cls_counts[C] = self.cls_counts.get(C,0) + self.word_counts[word]

        self.ordered_cls_list = list(self.cls_counts.keys())

    def display_summary(self):
        return """Using Brown Clusters with %d cluster and %d word forms"""%(len(self.ordered_cls_list),len(self.w2cls))
        
    def cls_list(self):
        """
        Returns an ordered list of clusters
        """
        return self.ordered_cls_list
         
    def __str__(self):
        return '\n'.join( ['P(%s|%d) = %f'%(w,C,self.word_emission_prob(w,logprob=False)) for w,C in self.w2cls.items()])
        
                
    def get_cls(self,wordform,defaultval='<UNK>'):
        """
        Returns the cluster to which this word belongs or a default
        value if this word is unknown to the clustering
        """
        return self.w2cls.get(wordform,defaultval)

    def word_emission_prob(self,wordform,logprob=True):
        """
        Returns P(w|C) if the word is known to the lexicon.
        Otherwise returns P(w=UNK|Unk cluster), that is 1.0. 
        
        @return P(w|C) that is the probability that this word is
        generated by its cluster C (which is automatically retrieved)
        
        @param wordform: the w of P(w|C)
        @param logprob:
        """
        C = self.get_cls(wordform,None)
        if C :
            N = self.cls_counts[C]
            w = self.word_counts[wordform]
            p = w/N
            return np.log(p) if logprob else p
        else:
            #if the word is not in a cluster, we assume it is part of the UNK cluster with only 1 element the UNK word 
            return 0.0 if logprob else 1.0
        
    def save_clusters(self,filename):
        """
        Saves the clusters in a json format
        """
        jfile = open(filename+'.json','w')
        jfile.write(json.dumps({'word_counts':self.word_counts,\
                                'w2cls':self.w2cls))
        jfile.close()
        
    def load_clusters(self,filename):
        """
        Loads the clusters from a json format
        """
        struct = json.loads(open(filename+'.json').read())
        return BrownLexicon(struct['w2cls'],struct['word_counts'],freq_threshold=0)
                
    @staticmethod
    def read_clusters(cls_filename,freq_thresh=1):
        """
        Builds a BrownLexicon object from raw path files.
        @param cls_filename: a path file produced by P. Liang Package
        @return
        """
        istream = open(cls_filename)
        clsIDs  = set([])
        word_counts = {}
        w2cls = {}
        for line in istream:
            ID, word, counts = line.split()
            clsIDs.add(ID)
            w2cls[word] = ID
            word_counts[word] = int(counts)
        istream.close()
        #now remap clsIDs to integers
        clsIDs = dict( [(C,idx) for (idx, C) in enumerate(clsIDs) ])
        w2cls  = dict( [(w,clsIDs[ID]) for (w,ID) in w2cls.items()] ) 
        return BrownLexicon(w2cls,word_counts,freq_thresh)
        

if __name__ == '__main__':
    blex = BrownLexicon.read_clusters("/Users/bcrabbe/parsing_as_LM/rnng/cls_example.txt",freq_thresh=1)
    print(blex)
