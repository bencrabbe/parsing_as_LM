#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

"""
Functions for plotting stats gathered when training the parser
"""

def plot_learningcurve(csvfilename,series_names=['overall','clusters','structure','struct_labels']):
    """
    Plots a learning curve file.
    """
    df = pd.read_table(csvfilename,sep=",",index_col=0)
    df.columns=series_names
    
    #plt.figure()
    axes = df.plot(title='Perplexities by subtasks (model = %s)'%('.'.join(csvfilename.split('.')[:-1]),))
    axes.set_xlabel("Number of trees processed (x1000)")
    axes.set_ylabel("Task perplexity")
    print(df)


if __name__ == '__main__':
    plt.style.use('ggplot')
    plot_learningcurve("example_LC.csv")
    plt.show()
