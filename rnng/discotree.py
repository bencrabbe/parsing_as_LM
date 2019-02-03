import sys
import os
import os.path

class DiscoTree:
    """
    That's a discontinuous phrase structure tree.
    With label, range (as a list of integers) and children.
    The I/O formats are compatible with van-cranenburgh's disco-dop.
    """
    def __init__(self,label,children = None ,child_index= -1):
        """
        A constructor to be used in a bottom-up fashion.
        Args :
           label        (str): the node label
           children:   (list): a list of DiscoTree nodes.
           child_index  (int): the position of the child in the string
        """
        self.label           = label
        if children :
            assert(child_index == -1)
            
            self.children    = children
            self.range       = [ ]
            for child in self.children:
                self.range.extend(child.range)
            self.range.sort()
            self.order_left_corner(recursive=False)
        else:
            assert(child_index >= 0 and children is None)
            self.range    = [child_index]
            self.children = [ ]
            
    def is_leaf(self):
        """
        Returns :
           Bool. True if this node is a leaf, false otherwise
        """
        return self.children == []

    def arity(self):
        return len(self.children)
            
    def __eq__(self,other):
        """
        Args:
           other (DiscoTree): the node with which to compare
        Returns:
            Bool. True if the two nodes share the same label and have the same range 
        """
        return self.label == other.label and self.range == other.range

    def __neq__(self,other):
        return not self.__eq__(other)

    def has_same_range(self,other_range):
        """
        Says if a node has equal range to some range
        Args:
           other_range (list): a list, iterable of integers
        Returns:
           True if both ranges are equal, false otherwise
        """
        return set(other_range) == set(self.range)

    def is_dominated_by(self,other_range):
        return set(self.range) <= set(other_range)
    
    def dominates(self,other_range,reflexive=True): 
        """ 
        Says if a range is dominated by this node or not.
        Args: 
           other_range  (list or set): a set of integers 
        Returns:
           bool. True if the range is fully dominated by this node as determined by a set inclusion test
        """
        return set(other_range) <= set(self.range) if reflexive else set(other_range) <  set(self.range)
   
    def get_lc_ancestors(self,other_range,min_range =-1):
        """
        Returns all the nodes in the tree whose range starts with the same
        left corner and that dominate this range. 

        Args:
           other_range (iterable) : a list or a set of integers
        Returns:
           A list of DiscoTree nodes 
        """
        res = [ ]

        if min_range == -1:
            min_range = min(other_range) 
 
        if self.left_corner() == min_range and self.dominates(other_range,reflexive=False):            
            res = [ self ]
             
        for child in self.children:
            res.extend(child.get_lc_ancestors(other_range,min_range))
             
        return res
    
    def gap_degree(self):
        """
        This computes the gap degree of this tree.
        Returns:
            (int) the gap degree of the subtree dominated by this node.
        """ 
        if self.is_leaf():
            return 0
        
        local_gd = sum([jdx != idx+1 for idx,jdx in zip(self.range,self.range[1:])])
        return max(local_gd,max([child.gap_degree() for child in self.children]))
    
    def gap_list(self):
        """
        This returns the list of gaps in the yield of this node.
        Returns:
            A list of couples (i,j) interpreted as a list of gaps
        """
        return [(idx,jdx) for idx,jdx in zip(self.range,self.range[1:])  if jdx != idx+1 ]

    def left_corner(self):
        """
        returns the left corner of this node
        """
        return self.range[0]

    def right_corner(self):
        """
        returns the right corner of this node
        """
        return self.range[-1]
    
    def cover(self):
        """
        Returns:
           a couple of int (i,j). This is the cover of the node (min(range),max(range)) 
        """
        return (self.left_corner(),self.right_corner())

    def order_left_corner(self,recursive=True):
        """
        Convenience in-place method that provides an arbitrary order to the nodes.
        The nodes are ordered according to a left corner convention.
        A child whose left corner < the left corner of another child precedes the other child.
        
        KwArgs:
           recursive (bool): if true, applies the method recursively
        """
        self.children.sort(key = lambda child: child.left_corner())
        if recursive:
            for child in self.children:
                child.order_left_corner(recursive)

    def covered_nodes(self,root):
        """
        Gets the nodes covered by this node using a left corner
        ordering. 
        Args:
          root  (DiscoTree) : the overall global root of this DiscoTree.
        Returns:
          the list of nodes covered by this node sorted according to the left corner convention.
        """ 
        MN = []
        for i,j in self.gap_list():
            MN.extend(root.max_nodes(i,j))
        cov_nodes = MN + self.children
        cov_nodes.sort(key = lambda c: c.left_corner())
        return cov_nodes
    
    def max_nodes(self,i,j):
        """
        Finds the set of maximal nodes covering a gap(i,j)
        Args: 
           i,j        (int). The gap interval
        Returns:
           a list of DiscoTree. The list of maximal nodes filling this gap.
        """
        res   = []
        ri,rj = self.cover()
        if (ri <= i and rj > i) or ( ri < j and rj >= j ): 
            for child in self.children:
                ci,cj = child.cover()
                if ci > i and cj < j:
                    #print('MAX',child.label)
                    res.append(child)
        #recursive additions
        for child in self.children:
            res.extend(child.max_nodes(i,j))
        return res

    #TRAVERSALS 
    def tokens(self,global_root=None,max_index=-1):
        """
        KwArgs:
           global_root (DiscoTree) :  the global root of the tree. No need to care in most cases
           max_index         (int) :  the index up to which the input is covered so far. No need to care in most cases
        Returns:
           a list. A list of DiscoTree objects, the leaves of the tree
        """
        if global_root is None:
                global_root = self
            
        if self.is_leaf():
                return [ self ]

        result = []
        for node in self.covered_nodes(global_root):
            if node.right_corner() > max_index:
                result.extend( node.tokens(global_root,max_index) )
                max_index = max(max_index,node.right_corner())  
        return result
         
    def words(self):
        """
        Returns:
           a list. A list of strings, the labels of the leaves of the tree 
        """
        toks = self.tokens()
        return [t.label for t in toks]

    def collect_nonterminals(self): 
        """ 
        Performs a traversal of the tree and returns the
        nonterminals labels according to this node ordering.

        Returns:
           a list. A list of strings, the labels of the leaves of the tree 
        """
        if self.is_leaf():
            return [ ]
        
        result = [ self.label ]
        for child in self.children:
            result.extend(child.collect_nonterminals())
        return result
    
    #TRANSFORMS
    def strip_tags(self):
        """
        In place (destructive) removal of pos tags
        """
        def gen_child(node):
            if len(node.children) == 1 and node.children[0].is_leaf():
                return node.children[0]
            return node
                
        self.children = [gen_child(child) for child in self.children]
        for child in self.children:
            child.strip_tags()
    
    def close_unaries(self,dummy_annotation='@'):
        """
        In place (destructive) unary closure of unary branches
        """
        if self.arity() == 1:
            current      = self
            unary_labels = []
            while current.arity() == 1 and not current.children[0].is_leaf():
                unary_labels.append(current.label)
                current = current.children[0]
            unary_labels.append(current.label)
            self.label = dummy_annotation.join(unary_labels)
            self.children = current.children
            
        for child in self.children:
            child.close_unaries()

    def expand_unaries(self,dummy_annotation='@'):
        """
        In place (destructive) expansion of unary symbols.
        """
        if dummy_annotation in self.label:
            unary_chain      = self.label.split(dummy_annotation)            
            current_label    = unary_chain.pop()
            current_children = self.children 
            while unary_chain:
                c = DiscoTree(current_label,self.children)
                current_label    = unary_chain.pop()
                current_children = [c]
            self.label    = current_label
            self.children = current_children
        
        for child in self.children:
            child.expand_unaries()

    #EVALUATION
    def eval_couples(self):
        """
        Extracts a list of evaluation couples from the tree,
        where each couple is (nonterminal,indexes of leaves in the yield of the nonterminal) 
        Returns:
           A list of eval couples. 
        """
        #look at the practice in the disco community
        cples = [(self.label,set(self.range))]
        for child in  self.children:
            cples.extend(child.eval_couples())
        return cples
            
    def compare(self,other):
        """
        Compares this tree to another and computes precision,recall,fscore.
        Assumes self is the reference tree, other is the predicted tree.
        
        Args:
          param other: the predicted tree
        Returns
           A tuple of floats. (precision,recall,fscore)
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

    #INPUT/OUTPUT
    def __str__(self):
        """
        Pretty prints the tree.
        Returns:
            A string. The pretty printing
        """
        self.order_left_corner(recursive=False)
        
        if self.is_leaf():
            return '%d=%s'%(list(self.range)[0],self.label)
        
        return '(%s %s)'%(self.label,' '.join([str(child) for child in self.children]))

    @staticmethod
    def read_tree(input_str):
        """
        Reads a one line s-expression using disco-dop format.
        This is a non robust function to syntax errors
        Args:
           Input_str  (str): a disco s-expr string
        Returns:
           A DiscoTree object
        """
        tokens = input_str.replace('(',' ( ').replace(')',' ) ').split()
        stack = [ ]
        for idx,tok in enumerate(tokens):
            if tok == '(':
                current = tokens[idx+1]
                stack.append(current)
            elif tok == ')':
                reduction = [ ]
                while type(stack[-1]) != str :
                    reduction.append(stack.pop())
                root_label = stack.pop()
                reduction.reverse()
                stack.append( DiscoTree(root_label,children = reduction) )
            else:
                if tokens[idx-1] != '(':
                    chunks = tok.split('=')
                    idx,wform = (chunks[0],'='.join(chunks[1:])) if len(chunks) > 2 else (chunks[0],chunks[1])
                    stack.append(DiscoTree(wform,child_index = int(idx)))
                    
        assert(len(stack) == 1)
        root = stack[-1]
        return root

    

if __name__ == "__main__":
    
    t = DiscoTree.read_tree('(U (X 1=Y 2=Z) (A 0=B 3=C 4=D))')
    static_oracle(t,t)
    print(t)
    
    t = DiscoTree.read_tree('(S (VP (VB 0=is) (JJ 2=rich)) (NP 1=John) (? 3=?))')
    print(t,'gap-degree',t.gap_degree())
    print(t.words())
    static_oracle(t,t)
    print()
    
    t2 = DiscoTree.read_tree("(ROOT (SBARQ (SQ (VP (WHADVP (WRB 0=Why)) (VB 4=cross) (NP (DT 5=the) (NN 6=road))) (VBD 1=did) (NP (DT 2=the) (NN 3=chicken))) (. 7=?)))")
    print(t2,'gap_degree',t2.gap_degree())
    print(t2.words())
    static_oracle(t2,t2)
    print()
    
    t3 = DiscoTree.read_tree('(S (NP 0=John) (VP (VB 1=is) (JJ 2=rich)) (PONCT 3=.))')
    print(t3,'gap_degree',t3.gap_degree())
    print(t3.words(),t3.compare(t3))
    
