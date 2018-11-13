# added to fix display issues
import matplotlib
matplotlib.use('Agg')

# credential for the static network analysis go to James Tollefson: https://www.kaggle.com/jamestollefson/enron-network-analysis

# basic modules
import numpy as np
import pandas as pd
import os
import argparse
import re
import csv
# network analysis
import networkx as nx
import nxviz as nv
# time analysis
from datetime import datetime
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#from scipy import stats
# for parallelization
import concurrent.futures
# functions loaded form other files
from functions.prepare import prepare_data # from folder functions import from the prepare file the function prepare_data
from functions.arcplot import make_arcplot
from functions.circosplot import make_circosplot
from functions.draw_network import draw
from functions.degree_centrality import dc
from functions.betweenness_centrality import bc
from functions.time_differences import td

########################################################################
# PARSING

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
    parser.add_argument("--time_difference", help='Make plot of time difference between interactions', action='store_true') # default is false
    args = parser.parse_args()

    data_link = args.data_link
    not_load = args.not_load
    arcplot = args.arcplot
    circosplot = args.circosplot
    draw_network = args.draw_network
    degree_centrality = args.degree_centrality
    betweenness_centrality = args.betweenness_centrality
    time_difference = args.time_difference

    return data_link, not_load, arcplot, circosplot, draw_network, degree_centrality, betweenness_centrality, time_difference

# pass on parser values
data_link, not_load, arcplot, circosplot, draw_network, degree_centrality, betweenness_centrality, time_difference = parser()

########################################################################
# DATA LOADING

# load data
if not_load:
    # load already prepared data
    data = pd.read_csv('./data/prepared/data.csv')

    print('data loaded')

else:

    # creating pool of processes for parallelization
    with concurrent.futures.ProcessPoolExecutor() as executor:

        for i, chunk in enumerate(pd.read_csv(data_link, chunksize=1000, iterator=True)):
            # process chunk
            future = executor.submit(prepare_data, chunk)
            result = future.result()
            # write csv
            result.to_csv('./data/prepared/chunk-{}.csv'.format(i))

    print('data prepared and saved')
'''
########################################################################
# PLOTTING

# create graph of data
#G = nx.from_pandas_edgelist(data, 'sender', 'recipient1', edge_attr=['date', 'subject'],)

# if wanted, make arcplot
if arcplot:
    make_arcplot(G)
    print('arcplot made')

# if wanted, make circosplot
if circosplot:
    make_circosplot(G)
    print('circosplot made')

# if wanted, draw network
if draw_network:
    draw(G)
    print('network drawn')

# if wanted, make plot of degree centrality
if degree_centrality:
    dc(G)
    print('degree centrality plotted')

# if wanted, make plot of betweenness centrality
if betweenness_centrality:
    bc(G)
    print('betweenness centrality plotted')

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


# if wanted, make plot of betweenness centrality
if time_difference:
    td(data['time_dif'])
    print('time differences plotted')

'''
# main function
if __name__ == '__main__':
    print('successful run')

