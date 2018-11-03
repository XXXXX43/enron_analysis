# added to fix display issues
import matplotlib
matplotlib.use('Agg')

# credential for the static network analysis go to James Tollefson: https://www.kaggle.com/jamestollefson/enron-network-analysis

import numpy as np
import pandas as pd
import os
import argparse
import re
import networkx as nx
import nxviz as nv
from functions.prepare import prepare_data # from folder functions import from the prepare file the function prepare_data
from functions.arcplot import make_arcplot
from functions.circosplot import make_circosplot
from functions.draw_network import draw
from functions.degree_centrality import dc
from functions.betweenness_centrality import bc



# parser function to collect and pass on values from terminal
def parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--print", help='Should I print?', type=bool, default=False)
    parser.add_argument("--data_link", help='Link to data that has to be loaded', type=str, default='./data/emails_kaggle.csv')
    parser.add_argument("--not_load", help='Prepare data from scratch?', action='store_false') # default is true
    parser.add_argument("--arcplot", help='Making arcplot of data', action='store_true') # default is false
    parser.add_argument("--circosplot", help='Making circosplot of data', action='store_true') # default is false
    parser.add_argument("--draw_network", help='Draw network', action='store_true') # default is false
    parser.add_argument("--degree_centrality", help='Make plot of degree centrality?', action='store_true') # default is false
    parser.add_argument("--betweenness_centrality", help='Make plot of betweenness centrality?', action='store_true') # default is false
    args = parser.parse_args()

    data_link = args.data_link
    not_load = args.not_load
    arcplot = args.arcplot
    circosplot = args.circosplot
    draw_network = args.draw_network
    degree_centrality = args.degree_centrality
    betweenness_centrality = args.betweenness_centrality

    return data_link, not_load, arcplot, circosplot, draw_network, degree_centrality, betweenness_centrality

# pass on parser values
data_link, not_load, arcplot, circosplot, draw_network, degree_centrality, betweenness_centrality = parser()

# load data
if not_load:
    # load already prepared data
    data = pd.read_csv('./data/emails_kaggle_prepared.csv')
else:
    # load unprepared data
    pd.options.mode.chained_assignment = None
    chunk = pd.read_csv(data_link, chunksize=500)
    data = next(chunk)

    # printing data information
    data.info()
    print(data.message[2])

    # prepare data
    data = prepare_data(data)

    print(data.head())

    # saving prepared data
    data.to_csv('./data/emails_kaggle_prepared.csv')

# create graph of data
G = nx.from_pandas_edgelist(data, 'sender', 'recipient1', edge_attr=['date', 'subject'],)

# if wanted, make arcplot
if arcplot:
    make_arcplot(G)

# if wanted, make circosplot
if circosplot:
    make_circosplot(G)

# if wanted, draw network
if draw_network:
    draw(G)

# if wanted, make plot of degree centrality
if degree_centrality:
    dc(G)

# if wanted, make plot of betweenness centrality
if betweenness_centrality:
    bc(G)

# main function
if __name__ == '__main__':
    print('Hello World')

