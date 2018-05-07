import sys
import os
import os.path
import re
from collections import Counter

class ConsTree:
    """
    That's your phrase structure tree.
    """
    def __init__(self,label,children=None):
        self.label = label
        self.children = [] if children is None else children

        
    def copy(self):
        """
        Performs a deep copy of this tree
        """
        return ConsTree(self.label,[c.copy() for c in self.children])
        
    def is_leaf(self):
        return self.children == []
    
    def add_child(self,child_node):
        self.children.append(child_node)
        
    def arity(self):
        return len(self.children)
        
    def get_child(self,idx=0):
        """
        @return the idx-th child of this node. 
        """
        return self.children[idx]

    def __str__(self):
        """
        Pretty prints the tree
        """
        return self.label if self.is_leaf() else '(%s %s)'%(self.label,' '.join([str(child) for child in self.children]))

    def tokens(self,labels=True):
        """
        @param labels: returns a list of strings if true else returns
        a list of ConsTree objects (leaves)
        @return the list of words at the leaves of the tree
        """
        if self.is_leaf():
                return [self.label] if labels else [self]
        else:
            result = []
            for child in self.children:
                result.extend(child.tokens(labels))
            return result
        
    def pos_tags(self):
        """
        @return the list of pos tags as ConsTree objects
        """
        if self.arity() == 1 and self.get_child().is_leaf():
            return [self]
        else:
            result = []
            for child in self.children:
                result.extend(child.pos_tags())
            return result
        
    def index_leaves(self):
        """
        Adds an numeric index to each leaf node
        """
        for idx,elt in enumerate(self.tokens(labels=False)):
            elt.idx = idx
            
    def triples(self):
        """
        Extracts a list of evalb triples from the tree
        (supposes leaves are indexed)
        """
        subtriples = []
        if self.is_leaf():
            return [(self.idx,self.idx+1,self.label)]

        for child in self.children:
                subtriples.extend(child.triples())
        leftidx  = min([idx for idx,jdx,label in subtriples])
        rightidx = max([jdx for idx,jdx,label in subtriples])
        subtriples.append((leftidx,rightidx,self.label))
        return subtriples

    def compare(self,other):
        """
        Compares this tree to another and computes precision,recall,
        fscore. Assumes self is the reference tree
        @param other: the predicted tree
        @return (precision,recall,fscore)
        """
        self.index_leaves()
        other.index_leaves()
        
        #filter out leaves
        #ref_triples  = set([(i,j,X) for i,j,X in self.triples() if j != i+1])
        #pred_triples = set([(i,j,X) for i,j,X in other.triples() if j != i+1])

        ref_triples  = set(self.triples())
        pred_triples = set(other.triples())
        
        intersect = ref_triples.intersection(pred_triples)
        isize = len(intersect)
        P = isize/len(pred_triples)
        R = isize/len(ref_triples)
        F = (2*P*R)/(P+R)
        return (P,R,F)
    
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

    def normalize_OOV(self,lexicon,unk_token):
        """
        Destructively replaces all leaves by the unk_token when the leaf label is not in
        lexicon. Normalizes numbers
        @param lexicon  : a set of strings
        @param unk_token: a string
        @return a pointer to the tree root
        """
        if self.is_leaf():
            if self.label not in lexicon:
                self.label = unk_token
        for child in self.children:
            child.normalize_OOV(lexicon,unk_token)
        return self

    def add_gold_tags(self,tag_sequence=None,idx=0):
        """
        Adds gold tags to the tree on top of leaves(for evalb compatibility).
        Destructive method.
        """
        newchildren = []
        for child in self.children:
            if child.is_leaf():
                label = tag_sequence[idx]
                tag = ConsTree(label,children=[child])
                newchildren.append(tag)
                idx += 1
            else:
                newchildren.append(child)
                idx = child.add_gold_tags(tag_sequence,idx)
        self.children=newchildren
        return idx
    
    def add_dummy_root(self,root_label='TOP'):
        """
        In place addition of a dummy root
        """
        selfcopy = ConsTree(self.label,children=self.children)
        self.children=[selfcopy]
        self.label = root_label
        
    def close_unaries(self,dummy_annotation='@'):
        """
        In place (destructive) unary closure of unary branches
        """
        if self.arity() == 1:
            current      = self
            unary_labels = []
            while current.arity() == 1 and not current.get_child().is_leaf():
                unary_labels.append(current.label)
                current = current.get_child()
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
            unary_chain = self.label.split(dummy_annotation)
            self.label  = unary_chain[0]
            backup      = self.children
            current     = self
            for label in unary_chain[1:]:
                c = ConsTree(label)
                current.children = [c] 
                current = c
            current.children = backup
            
        for child in self.children:
            child.expand_unaries()

    def left_markovize(self,dummy_annotation=':'):
        """
        In place (destructive) left markovization (order 0)
        """
        if len(self.children) > 2:
            left_sequence = self.children[:-1]
            dummy_label = self.label if self.label[-1] == dummy_annotation else self.label+dummy_annotation
            dummy_tree = ConsTree(dummy_label, left_sequence)
            self.children = [dummy_tree,self.children[-1]]
        for child in self.children:
            child.left_markovize()

    def right_markovize(self,dummy_annotation=':'):
        """
        In place (destructive) right markovization (order 0)
        """
        if len(self.children) > 2:
            right_sequence = self.children[1:]
            dummy_label = self.label if self.label[-1] == dummy_annotation else self.label+dummy_annotation
            dummy_tree = ConsTree(dummy_label, right_sequence)
            self.children = [self.children[0],dummy_tree]
        for child in self.children:
            child.right_markovize()

    def unbinarize(self,dummy_annotation=':'):
        """
        In place (destructive) unbinarization
        """
        newchildren = []
        for child in self.children:
            if child.label[-1] == dummy_annotation:
                child.unbinarize()
                newchildren.extend(child.children)
            else:
                child.unbinarize()
                newchildren.append(child)
        self.children = newchildren

    def collect_nonterminals(self):
        """
        Returns the list of nonterminals found in a tree:
        """
        if not self.is_leaf():
            result =  [self.label]
            for child in self.children:
                result.extend(child.collect_nonterminals())
            return result
        return []

    @staticmethod
    def read_tree(input_str):
        """
        Reads a one line s-expression.
        This is a non robust function to syntax errors
        @param input_str: a s-expr string
        @return a ConsTree object
        """
        tokens = input_str.replace('(',' ( ').replace(')',' ) ').split()
        stack = [ConsTree('dummy')]
        for idx,tok in enumerate(tokens):
            if tok == '(':
                current = ConsTree(tokens[idx+1])
                stack[-1].add_child(current)
                stack.append(current)
            elif tok == ')':
                stack.pop()
            else:
                if tokens[idx-1] != '(':
                    stack[-1].add_child(ConsTree(tok))
        assert(len(stack) == 1)
        return stack[-1].get_child()


class PennTreebank:
    """
    That's a namespace for processing penn treebank sources.
    This is meant to generate the traditional setup for PS parsing
    """
    @staticmethod
    def flatten_file(istream):
        """
        Reads a PTB-III file and turns the trees in one-line formatted s-expressions
        @param istream: the input stream to a ptb3 mrg file
        @return a list of strings with one line s-expressions coding the trees.
        """
        trees = []
        bfr = ''
        copen  = 0
        for line in istream:
            for c in line:
                if c == '(':
                    copen += 1
                elif c == ')':
                    copen -= 1
            bfr += line
            if copen == 0 and not bfr.isspace():
                trees.append(' '.join(bfr.split()))
                bfr = ''
        assert(not bfr or bfr.isspace()) #checks that we do not forget stuff
        return trees
    
    
    @staticmethod
    def preprocess_file(filename,strip_tags,close_unaries):
        """
        That's a generator function yielding tree objects.
        @yield ConsTree objects
        """
        ifile = open(filename)
        trees =  PennTreebank.flatten_file(ifile)
        for t_string in trees:
            tree = ConsTree.read_tree(t_string).children[0]
            tree.add_dummy_root()
            
            PennTreebank.normalize_numbers(tree,num_token='<num>')
            PennTreebank.strip_traces(tree)
            PennTreebank.strip_decoration(tree)
            if strip_tags:
                ConsTree.strip_tags(tree)
            if close_unaries:
                ConsTree.close_unaries(tree)
            yield tree
        ifile.close()

    @staticmethod
    def strip_decoration(ctree):
        """
        That's an inplace destructive function removing PTBIII trees node label decorations (such as functions etc).
        @param ctree: a ConsTree object.
        """
        if not (len(ctree.children) == 1 and ctree.get_child().is_leaf()):#if this is a tag stop recursion
            ctree.label = ctree.label.split('-')[0]
            ctree.label = ctree.label.split('=')[0]
            for child in ctree.children:
                PennTreebank.strip_decoration(child)
        
    @staticmethod
    def strip_traces(ctree):
        """
        That's an inplace destructive function removing traces from PTBIII trees.
        @param ctree: a ConsTree object.
        @return true if the subtree is to be removed, false otherwise
        """
        if ctree.label == '-NONE-':
            return True
        if not ctree.children:
            return False
        
        removals = [PennTreebank.strip_traces(child) for child in ctree.children]

        if all(removals):
            return True
        else:
            ctree.children = [child for flag,child in zip(removals,ctree.children) if not flag]
            return False

    @staticmethod
    def count_categories(treebankfile):
        """
        Counts the occurrences of nonterminal categories and returns
        the result as a counter
        @return a Counter
        """
        def rec_count(tree_node,counter):
            counter[tree_node.label] += 1
            for child in tree_node.children:
                if not child.is_leaf():
                    rec_count(child,counter)

        treebankstream = open(treebankfile)
        c = Counter()
        for line in treebankstream:
            T = ConsTree.read_tree(line)
            T.strip_tags()
            rec_count(T,c)
        treebankstream.close()
        return c
            
    @staticmethod
    def preprocess_src_dir(dirpath,strip_tags,close_unaries):
        """
        A path to a PTB mrg directory.
        @param dirpath: the path
        @return a tree list
        """
        files = os.listdir(dirpath)
        tree_list = []
        for f in files:
            filename = str(os.path.join(dirpath,f))
            tree_list.extend(PennTreebank.preprocess_file(filename,strip_tags,close_unaries))
        return tree_list

    @staticmethod
    def generate_standard_split(ptb_root,out_data_dir):
        """
        A path to a PTB mrg directory.
        @param ptb_root: the PTB root dir (dominates the 00, 01, ... 24 dirs)
        @param out_data_dir: the dir where to write the files
        """
        print('Generating PTB data...',file=sys.stderr)
        train_dirs = ['0%d'%(elt,) for elt in range(2,10)] + [str(elt) for elt in range(10,22)]
        dev_dir    = str(24)
        test_dir   = str(23)

        train_file = open(os.path.join(out_data_dir,'ptb_train.mrg'),'w')
        dev_file   = open(os.path.join(out_data_dir,'ptb_dev.mrg'),'w')
        test_file  = open(os.path.join(out_data_dir,'ptb_test.mrg'),'w')
        train_raw  = open(os.path.join(out_data_dir,'ptb_train.raw'),'w')
        dev_raw    = open(os.path.join(out_data_dir,'ptb_dev.raw'),'w')
        test_raw   = open(os.path.join(out_data_dir,'ptb_test.raw'),'w')
        
        for d in train_dirs:
            print('Processing wsj-%s'%(d,),file=sys.stderr)
            print('\n'.join([str(t) for t in PennTreebank.preprocess_src_dir(str(os.path.join(ptb_root,d)),strip_tags=False,close_unaries=False)]),file=train_file)
            print('\n'.join([' '.join(t.tokens(labels=True)) for t in PennTreebank.preprocess_src_dir(str(os.path.join(ptb_root,d)),strip_tags=False,close_unaries=False)]),file=train_raw)

            
        print('Processing wsj-%s'%(dev_dir,),file=sys.stderr)
        devtrees = PennTreebank.preprocess_src_dir(str(os.path.join(ptb_root,dev_dir)),strip_tags=False,close_unaries=False)
        print('\n'.join([str(t) for t in devtrees]),file=dev_file)
        print('\n'.join([' '.join(t.tokens(labels=True)) for t in devtrees]),file=dev_raw)
        print('Processing wsj-%s'%(test_dir,),file=sys.stderr)
        testtrees = PennTreebank.preprocess_src_dir(str(os.path.join(ptb_root,test_dir)),strip_tags=False,close_unaries=False)
        print('\n'.join([str(t) for t in testtrees]),file=test_file)
        print('\n'.join([' '.join(t.tokens(labels=True)) for t in testtrees]),file=test_raw)

        train_file.close()
        dev_file.close()
        test_file.close()
        train_raw.close()
        dev_raw.close()
        test_raw.close()

    @staticmethod
    def normalize_numbers(tree,num_token='<num>'):
        """
        Replaces all numbers with the <num> token
        """
        if tree.is_leaf():
            if re.match(r'[0-9]+([,/\.][0-9]+)*',tree.label):
                tree.label = num_token
        for child in tree.children:
            PennTreebank.normalize_numbers(child,num_token)
        return tree
                
if __name__ == '__main__':
    #Generates Penn TB with classical setup
    #PennTreebank.generate_standard_split('/data/Corpus/ptb/treebank_3/parsed/mrg/wsj','/home/bcrabbe/parsing_as_LM/rnng')
    PennTreebank.generate_standard_split('/Users/bcrabbe/Desktop/ptb/treebank_3/parsed/mrg/wsj','/Users/bcrabbe/parsing_as_LM/rnng')

    c = PennTreebank.count_categories('ptb_train.mrg')
    for cat,count in c.items():
        print('%s :: %d'%(cat,count))
