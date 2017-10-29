class DependencyTree:

    def __init__(self,tokens=None, edges=None):
        #pb ensure there is one and only one root at init
        self.edges  = [] if edges is None else edges   #couples (gov_idx,dep_idx)
        self.tokens = ['#ROOT#'] if tokens is None else tokens #list of wordforms
    
    def __str__(self):
        gdict = dict([(d,g) for (g,d) in self.edges])
        return '\n'.join(['\t'.join([str(idx+1),tok,str(gdict[idx+1])]) for idx,tok in enumerate(self.tokens[1:])])

    def is_projective(self,root=0,ideps=None):
        """
        Tests if a dependency tree is projective.
        @param root : the index of the root node
        @param ideps: a dict index -> list of immediate dependants.
        @return: (a boolean stating if the root is projective , a list
        of children idxes)
        """
        if ideps is None:#builds dict if not existing
            ideps = {}
            for gov,dep in self.edges:
                if gov in ideps:
                    ideps[gov].append(dep)
                else:
                    ideps[gov] = [dep]
        
        allc = [root]                              #reflexive
        if root not in ideps:                      #we have a leaf
            return (True,allc)
        for child in ideps[root]:
            proj,children = self.is_projective(child,ideps)
            if not proj:
                return (proj,None)
            allc.extend(children)                   #transitivity

        allc.sort()                                 #checks projectivity
        for _prev,_next in zip(allc,allc[1:]):
            if _next != _prev+1:
                return (False,None)
        return (True, allc)
    
    @staticmethod
    def read_tree(istream):
        """
        Reads a tree from an input stream in CONLL-U format.
        Currently ignores empty nodes and compound spans.
        
        @param istream: the stream where to read from
        @return: a DependencyTree instance 
        """
        deptree = DependencyTree()
        bfr = istream.readline()
        while bfr:
            if bfr[0] == '#':
                bfr = istream.readline()
            elif (bfr.isspace() or bfr == ''):
                if deptree.N() > 0:
                    return deptree
                else:
                    bfr = istream.readline()
            else:
                line_fields = bfr.split()
                idx, word, governor_idx = line_fields[0],line_fields[1],line_fields[6]
                if not '.' in idx: #empty nodes have a dot (and are discarded here)
                    deptree.tokens.append(word)
                    deptree.edges.append((int(governor_idx),int(idx)))
                bfr = istream.readline()
        return None

    
    def accurracy(self,other):
        """
        Compares this dep tree with another by computing their UAS.
        Assumes this tree is the reference tree
        @param other: other dep tree
        @return : the UAS as a float
        """
        S1 = set(self.edges)
        S2 = set(other.edges)
        return len(S1.intersection(S2)) / len(S1)
    
    def N(self):
        """
        Returns the length of the input
        """
        return len(self.tokens)
    
    def __getitem__(self,idx):
        """
        Returns the token at index idx
        """
        return self.tokens[idx]
        

#reads UD style treebanks cleaned up data sets
def UDtreebank_reader(filename,tokens_only=True):
    treebank = []
    istream = open(filename)
    dtree = DependencyTree.read_tree(istream)
    while dtree != None:
        if dtree.is_projective():
            if tokens_only:
                treebank.append(dtree.tokens)
            else:
                treebank.append(dtree)
        else:
            print("Skipped non projective tree")
        dtree = DependencyTree.read_tree(istream)
    istream.close()
    return treebank

#reads Mikolov PTB cleaned up data sets
def ptb_reader(filename):
    istream = open(filename)
    treebank = []
    for line in istream:
        treebank.append(line.split())
    istream.close()
    return treebank

if __name__ == '__main__':
    #example code
    pass
