import sys
import os
import os.path

"""
@see http://research.nii.ac.jp/~kanazawa/mcfgplus/2011/20110909-MCFG-upload.pdf
for a reminder on MCFG.

@see Kanazawa excellent lecture notes for another reminder : http://research.nii.ac.jp/~kanazawa/FormalGrammar/lecture5.pdf

@see http://discodop.readthedocs.io/en/latest/
for a reminder on disco-dop

@see Maier and Lichte
https://pdfs.semanticscholar.org/0e66/89f446894141359b6753b9a069742eece81d.pdf
"""
class DiscoTree:
    """
    That's a discontinuous phrase structure tree.
    With label, yield_range (as a list of integers) and children.
    The I/O formats are compatible with van-cranenburgh's disco-dop.
    """
    def __init__(self,label,yield_range=None,children=None):
        """
        @param label: a string
        @param yield_range: a list of integers
        @param children: a list of DiscoTree nodes.
        """
        self.label       = label
        self.yield_range = [] if yield_range is None else yield_range
        self.children    = [] if children is None else children
        
    def is_leaf(self):
        return self.children == []

    def add_child(self,child_node):
        self.children.append(child_node)

    def __eq__(self,other):
        """
        Node equality: two nodes are equal iff they share same label
        and span.
        @return a bool
        """
        #print('eq',self.yield_range,other.yield_range)
        return self.label == other.label and self.yield_range == other.yield_range
        
        
    def rank(self):
        """
        This computes the LCFRS rank of the subtree dominated by this node
        """
        if self.is_leaf():
            return 0
        local_rank = len(self.children)
        return max(local_rank,max([child.rank() for child in self.children]))

    def gap_degree(self):
        """
        This computes the gap degree of this tree.
        This supposes the yield_range of each node to be properly sorted (!)
        @return the gap degree of the tree dominated by this node.
        """
        if self.is_leaf():
            return 0
        local_gd = sum([jdx != idx+1 for idx,jdx in zip(self.yield_range,self.yield_range[1:])])
        return max(local_gd,max([child.gap_degree() for child in self.children]))

    def gap_list(self):
        """
        This returns the list of gaps in the yield of this node.
        This supposes the yield_range of each node to be properly sorted (!)
        @return a list of couples (i,j) interpreted as open intervals ]i,j[ 
        """
        res = []
        for idx,jdx in zip(self.yield_range,self.yield_range[1:]):
            if jdx != idx+1:
                res.append((idx,jdx))
        return res

    def left_corner(self):
        """
        returns the left corner of this node
        """
        return min(self.yield_range)

    def right_corner(self):
        """
        returns the right corner of this node
        """
        return max(self.yield_range)
    
    def range(self):
        """
        @return the (discontinuous) yield range of this node as a couple (i,j) 
        """
        return (min(self.yield_range),max(self.yield_range))

    
    def max_node(self,i,j,parent=None):
        """
        Finds the minimum set of nodes covering a gap between (i,j)
        @param i,j the open interval ]i,j[
        @param parent the parent of the processed node 
        @return the (list) of nodes filling the gap
        @see Maier and Lichte 2011, (Maximal node)
        """
        if parent:
            a,b =  self.range()
            #print(self.label,a,b)
            if i < a and b < j and any([idx <= i or idx >= j for idx in parent.yield_range]):
                return [self]
        res = []
        for child in self.children:
            res.extend(child.max_node(i,j,parent=self))
        return res

                    
    def get_child(self,idx=0):
        """
        @return the ith child of this node.
        No ordering is guaranteed, use it with care or use order_children earlier on.
        """
        return self.children[idx]

    def order_children(self):
        """
        Convenience in-place method that provides an arbitrary order to the nodes.
        It attempts to order them in such a way that the tree is
        easier to read if pretty printed.
        """
        self.children.sort(key=lambda x:x.yield_range[0])
    
    def __str__(self):
        """
        Pretty prints the tree.
        """
        if self.is_leaf():
            return '%d=%s'%(self.yield_range[0],self.label)
        
        self.order_children()
        return '(%s %s)'%(self.label,' '.join([str(child) for child in self.children]))

    def tokens(self):
        """
        Warning, ordering is not guaranteed.
        @return the list of words at the leaves of the tree as a list of DiscoTree objects.
        """
        if self.is_leaf():
                return [self]
        else:
            result = []
            for child in self.children:
                result.extend(child.tokens())
            return result
        
    def words(self):
        """
        @return the list of words at the leaves of the tree as a list of strings.
        """
        toks = self.tokens()
        toks.sort(key=lambda x:x.yield_range[0])
        return [t.label for t in toks]
                
    def eval_couples(self):
        """
        Extracts a list of evaluation couples from the tree,
        where each couple is (nonterminal,indexes of leaves in the yield of the nonterminal) 
        @return a list of eval couples
        """
        #look at the practice in the disco community
        cples = [(self.label,set(self.yield_range))]
        for child in  self.children:
            cples.extend(child.eval_couples())
        return cples
            
    def compare(self,other):
        """
        Compares this tree to another and computes precision,recall,fscore.
        Assumes self is the reference tree, other is the predicted tree.
        
        @param other: the predicted tree
        @return (precision,recall,fscore)
        """
        cplesA = self.eval_couples()
        cplesB = other.eval_couples()
        AcapB  = [ c for c in cplesA if c in cplesB ]
        refN   = len(cplesA)
        predN  = len(cplesB)
        interN = len(AcapB)
        prec   = interN/predN
        rec    = interN/refN
        fsc    = 2*prec*rec/(prec+rec)
        return prec,rec,fsc

    
    def index_tree(self):
        """
        Internal method that ensures that the node ranges are properly filled
        """
        if self.is_leaf():
            return self.yield_range
        
        self.yield_range = []
        for child in self.children:
            self.yield_range.extend(child.index_tree())
        self.yield_range.sort()
        return self.yield_range
            
    @staticmethod
    def read_tree(input_str):
        """
        Reads a one line s-expression using disco-dop format.
        This is a non robust function to syntax errors
        @param input_str: a s-expr string
        @return a DiscoTree object
        """
        tokens = input_str.replace('(',' ( ').replace(')',' ) ').split()
        stack = [DiscoTree('dummy')]
        for idx,tok in enumerate(tokens):
            if tok == '(':
                current = DiscoTree(tokens[idx+1])
                stack[-1].add_child(current)
                stack.append(current)
            elif tok == ')':
                stack.pop()
            else:
                if tokens[idx-1] != '(':
                    idx,wform = tok.split('=')
                    stack[-1].add_child(DiscoTree(wform,yield_range=[int(idx)]))
                    
        assert(len(stack) == 1)
        root = stack[-1].get_child()
        root.index_tree()
        return root

    
if __name__ == "__main__":
    t = DiscoTree.read_tree('(S (VP (VB 0=is) (JJ 2=rich)) (NP 1=John) (? 3=?))')
    print(t,'gap-degree',t.gap_degree(),'rank',t.rank())
    print(t.words())
    print()
    t2 = DiscoTree.read_tree("(ROOT (SBARQ (SQ (VP (WHADVP (WRB 0=Why)) (VB 4=cross) (NP (DT 5=the) (NN 6=road))) (VBD 1=did) (NP (DT 2=the) (NN 3=chicken))) (. 7=?)))")
    print(t2,'gap_degree',t2.gap_degree(),'rank',t2.rank())
    print(t2.words())
    print()
    
    t3 = DiscoTree.read_tree('(S (NP 0=John) (VP (VB 1=is) (JJ 2=rich)) (PONCT 3=.))')
    print(t3,'gap_degree',t3.gap_degree(),'rank',t3.rank())
    print(t3.words(),t3.compare(t3))
    
