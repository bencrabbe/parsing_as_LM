#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

"""
Plots measures from the parser measures output
"""


def plot_measures(csvfilename=None,datatable=None,select=[True,True,True],labels=None,row_indexes=None,label_rotation=45):
    """
    Plots up to two measures for a given text.
    
    @param csvfilename: file where to find the data
    @param datatable  : panda data frame where to find the data
    @param select     : selects measures to plot [UnkWord,Surprisal,StructuralComplexity] at most two can be selected
    @param labels     : labels for the measures on the plot
    @param row_indexes: a list of integer indexes that allows (discontinuous) subsetting of the data
    """
    assert((not csvfilename is None) or (not datatable is None))
    assert(sum(select) <= 2)
    assert(len(labels) == sum(select))

    if csvfilename:
        df = pd.read_table(csvfilename,sep=",",index_col=0)
    elif datatable:
        df = datatable.copy()
        
    #col subsetting
    sub_idxes = [0]+[ idx+1 for idx, flag in enumerate(select) if flag]
    print(sub_idxes)
    df        = df[df.columns[sub_idxes]]
    df.columns=['tokens']+labels
    
    #row subsetting
    df = df.iloc[row_indexes]
    df.set_index("tokens",drop=True,inplace=True)
    print(df)
    if len(labels) > 1:
        axes = df.plot(title=' and '.join(labels)+' per word',secondary_y=labels[1],mark_right=False,kind='bar',rot=label_rotation)    
        axes.set_ylabel('%s scale'%(labels[0]))
        axes.right_ax.set_ylabel('%s scale'%(labels[1]))
    else:
        axes = df.plot(title=labels[0]+' per word',kind='bar',rot=label_rotation)
        axes.set_ylabel(labels[0])
    axes.set_xlabel('')
    
if __name__ == '__main__':
    plt.style.use('ggplot')
    plot_measures(csvfilename='exemple_measures.csv',select=[False,True,True],labels=['Surprisal','Complexity'],row_indexes=list(range(13,25)))
    plt.show()
