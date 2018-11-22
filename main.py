########################################################################
#TO DO

'''
- prepare.py: bug if no correspondent found in get_address
- remove < in text (forwarded messages)
'''

########################################################################
# IMPORT

# added to fix display issues
import matplotlib
matplotlib.use('Agg')

# basic modules
import numpy as np
import pandas as pd
import os
import argparse
import re
import csv
import glob
# text processing
from textblob import TextBlob
from nltk.tokenize import TabTokenizer
from textblob.sentiments import NaiveBayesAnalyzer # for sentiment analysis
# showing progress
from tqdm import tqdm
# network analysis
import igraph as ig
# time analysis
from datetime import datetime
# parallelization
#from multiprocessing import cpu_count, Pool
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# functions loaded form other files
from functions.processing import process_text # from folder functions import from the prepare file the function prepare_data
from functions.degree_distro import degree_distro


########################################################################
# PARSING

# parser function to collect and pass on values from terminal
def parser():
    parser = argparse.ArgumentParser(description='Specify analysing process')
    parser.add_argument("--data_link", help='Link to data that has to be loaded', \
        type=str, default='./data/prepared/total.txt')
    parser.add_argument("--cols", help='Which columns of data to use: sender, recipient, subject, text, date, conv_partner, sentiment?', \
        type=str, default='sender recipient subject date sentiment')
    parser.add_argument("--stat", help='Calculate statistics', \
        action='store_true') # default is false
    parser.add_argument("--not_load", help='Prepare data from scratch?', \
        action='store_false') # default is true
    parser.add_argument("--draw_network", help='Draw network', action='store_true') # default is false
    parser.add_argument("--distro", help='Degree distribution', action='store_true') # default is false
    parser.add_argument("--circos_plot", help='Make circos plot', action='store_true') # default is false

    args = parser.parse_args()

    data_link = args.data_link
    not_load = args.not_load
    draw_network = args.draw_network
    cols = args.cols.split()
    circos_plot = args.circos_plot
    stat = args.stat
    distro = args.distro

    return data_link, not_load, draw_network, cols, circos_plot, stat, distro

# pass on parser values
data_link, not_load, draw_network, cols, circos_plot, stat, distro = parser()


########################################################################
# PARALLELIZATION

#nCores = cpu_count() #number of CPU cores
#partitions = 30 #number of partitions


########################################################################
# DATA LOADING

# load data
if not_load:
    data = pd.read_csv(data_link, header=0, sep=' ', usecols=cols)
    print('data loaded')

else:

    for i, chunk in tqdm(enumerate(pd.read_csv('./data/emails_kaggle.csv', chunksize=1000, \
            iterator=True)), total=518, unit='chunks'):
        # assigninig new columns to processed information
        chunk['sender'], chunk['recipient'], chunk['subject'], chunk['text'], \
            chunk['date'], chunk['conv_partner'], chunk['sentiment'] = zip(*chunk['message'].map(process_text))
        # drop former columns
        chunk = chunk.drop(['file', 'message'], axis=1)
        # removing e-mails not containing necessary information
        chunk = chunk.dropna()

        # write csv
        chunk.to_csv('./data/prepared/chunk-{}.txt'.format(i), header=True, sep=' ')

    data = pd.concat([pd.read_csv(f, header=0, sep=' ') for f in glob.glob('./data/prepared/chunk-*.txt')])
    print('loaded')
    data = data.sort_values(by=['date'])
    print('sorted')
    data.to_csv('./data/prepared/total.txt', header=True, sep=' ')

    print('data prepared and saved')


########################################################################
# MAKING GRAPH

# create graph of data
g = ig.Graph.TupleList(data[['sender', 'recipient']].itertuples(index=False), weights=False)

print('graph made')


########################################################################
# STATISTICS

if stat:
    # calculate and print statistics for each column
    for col in data.columns:
        data[col].describe().to_csv("./statistics/{}.csv".format(col))

if distro:
    degree_distro(g)

    print('degree distribution plotted')


########################################################################
# PLOTTING

# if wanted, draw network with communities
if draw_network:
    layout = g.layout('fr') # Fruchterman-Reingold force-directed algorithm
    communities = g.community_multilevel()
    ig.plot(communities, './figures/communities.png', mark_groups=True, vertex_label=None, layout=layout, vertex_size=2, edge_arrow_size=.2)

    print('network drawn')

if circos_plot:
    layout = g.layout_circle()
    ig.plot(g, './figures/circos.png', \
        vertex_label=None, vertex_size=2, edge_arrow_size=.2, layout=layout)

    print('circos plot made')


'''
########################################################################
# TIME ANALYSIS

data = data.sort_values(by=['recipient1', 'recipient2'])

# add time differenc column
date_format = "%Y-%m-%d %H:%M:%S" # date format
data['date'] = pd.to_datetime(data['date'], format=date_format) # new date column
data['time_dif'] = data['date'] - data['date'].shift(1) # time difference column next e-mail
data['time_dif'] = data['time_dif'].apply(lambda x: x.total_seconds()) # time differences in seconds
data['time_dif'] = data['time_dif'].fillna(0)


#sns.distplot(data['time_dif'])




########################################################################
# SENTIMENT ANALYSIS



'''
########################################################################
# MAIN
if __name__ == '__main__':
    print('successful run')
