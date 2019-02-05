########################################################################
# IMPORT

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

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
from sklearn.preprocessing import normalize
from itertools import combinations
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
from dateutil.relativedelta import relativedelta
import time
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import norm
import matplotlib.mlab as mlab
from igraph.drawing.text import TextDrawer
import cairo
# functions loaded form other files
from functions.degree_distro import degree_distro
from functions.degree_centrality import dc
from functions.preprocessing import prepare_data
from functions.sentiment_analysis import sentiment_analysis
from functions.interaction_time import interaction_time
from functions.activity import activity_plot
from functions.measure_changes import measure_change
from functions.plot_routines import lineplot, boxplot
from functions.dynamics import dynamics_analysis


########################################################################
# PARSING

# parser function to collect and pass on values from terminal
def parser():
    parser = argparse.ArgumentParser(description='Specify analysing process')
    parser.add_argument("--data_link", help='Link to data that has to be loaded', \
        type=str, default='./data/prepared/total.txt')
    parser.add_argument("--cols", help='Which columns of data to use: sender, recipient, recipient2, recipient3, subject, text, date, conv_partner, sentiment?', \
        type=str, default='sender recipient recipient2 recipient3 recipient4 recipient5 date sentiment')
    parser.add_argument("--not_load", help='Prepare data from scratch?', \
        action='store_false') # default is true
    parser.add_argument("--draw_static", help='Draw network with communities for all times (circos and network like plot)', action='store_true') # default is false
    parser.add_argument("--draw_dynamic", help='Draw graph for each month', action='store_true') # default is false
    parser.add_argument("--distro", help='Degree distribution', action='store_true') # default is false
    parser.add_argument("--centrality", help='Show degree centrality', action='store_true') # default is false
    parser.add_argument("--sentiment", help='Plot sentiment distribution', action='store_true') # default is false
    parser.add_argument("--interaction", help='Plot interaction time differences', action='store_true') # default is false
    parser.add_argument("--activity", help='Plot e-mail activity over time', action='store_true') # default is false
    parser.add_argument("--detect_change", help='Detect changes in network measures', action='store_true') # default is false
    parser.add_argument("--dynamic", help='Analysis dynamic case', action='store_true') # default is false

    args = parser.parse_args()

    data_link = args.data_link
    not_load = args.not_load
    draw_static = args.draw_static
    draw_dynamic = args.draw_dynamic
    cols = args.cols.split()
    distro = args.distro
    centrality = args.centrality
    sentiment = args.sentiment
    interaction = args.interaction
    activity = args.activity
    detect_change = args.detect_change
    dynamic = args.dynamic

    return data_link, not_load, draw_static, draw_dynamic, cols, distro, centrality, sentiment, interaction, activity, detect_change, dynamic


# pass on parser values
data_link, not_load, draw_static, draw_dynamic, cols, distro, centrality, sentiment, interaction, activity, detect_change, dynamic = parser()


########################################################################
# DATA LOADING

# load data
if not_load:
    data = pd.read_csv(data_link, header=0, sep=' ', usecols=cols)#.head(10000)
    print('data loaded')

else:
    for i, chunk in tqdm(enumerate(pd.read_csv('./data/emails_kaggle.csv', chunksize=1000, \
            iterator=True)), total=518, unit='chunks'):

        # preparing data
        result = prepare_data(chunk)

        # write csv
        result.to_csv('./data/prepared/chunk-{}.txt'.format(i), sep=' ')

    data = pd.concat([pd.read_csv(f, header=0, sep=' ') for f in glob.glob('./data/prepared/chunk-*.txt')])
    print('loaded')
    data = data.sort_values(by=['date'])
    print('sorted')
    data.to_csv('./data/prepared/total.txt', header=True, sep=' ')

    print('data prepared and saved')

# convert date
date_format = "%Y-%m-%d %H:%M:%S" # date format
data['date'] = pd.to_datetime(data['date'], format=date_format)
# converting to timestamp
data['date'] = data[['date']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)
# focus on data of certain time range (1998-10-1 to 2002-07-31)
start = datetime(1998, 10, 1)
end = datetime(2002, 8, 1)


########################################################################
# MAKING GRAPH

# add column for weights
data['weights'] = 1

# maximum of 5 recipients in dataset; weights for edges decrease depending on position of recipient
# recipient
data1 = data.copy()
data1.drop(['recipient2', 'recipient3', 'recipient4', 'recipient5'], axis=1, inplace=True)
# recipient2
data2 = data.copy()
data2.drop(['recipient', 'recipient3', 'recipient4', 'recipient5'], axis=1, inplace=True)
data2.dropna(subset=['recipient2'], inplace=True)
data2.rename(index=str, columns={'recipient2':'recipient'}, inplace=True)
data2['weights'] = 0.8
# recipient3
data3 = data.copy()
data3.drop(['recipient', 'recipient2', 'recipient4', 'recipient5'], axis=1, inplace=True)
data3.dropna(subset=['recipient3'], inplace=True)
data3.rename(index=str, columns={'recipient3':'recipient'}, inplace=True)
data3['weights'] = 0.6
# recipient4
data4 = data.copy()
data4.drop(['recipient', 'recipient2', 'recipient3', 'recipient5'], axis=1, inplace=True)
data4.dropna(subset=['recipient4'], inplace=True)
data4.rename(index=str, columns={'recipient4':'recipient'}, inplace=True)
data4['weights'] = 0.4
# recipient5
data5 = data.copy()
data5.drop(['recipient', 'recipient2', 'recipient3', 'recipient4'], axis=1, inplace=True)
data5.dropna(subset=['recipient5'], inplace=True)
data5.rename(index=str, columns={'recipient5':'recipient'}, inplace=True)
data5['weights'] = 0.2

# join data
frames = [data1, data2, data3, data4]
data = pd.concat(frames, axis=0, join='outer', ignore_index=True)

# create graph of data
g = ig.Graph.TupleList(data[['sender', 'recipient', 'weights', 'date', 'sentiment']].itertuples(index=False), directed=True, edge_attrs=['weight', 'date', 'sentiment'])

# graph info
num_vertices = len(g.vs) # number of vertices
num_links = len(g.es)
print('graph made: {} nodes, {} links'.format(num_vertices, num_links))


########################################################################
# STATISTICS

if distro:
    # simplify graph by removing loops and multiple edges; weights are added
    g_ = g.simplify(multiple=True, combine_edges=dict(weight='sum', date='ignore', sentiment='ignore'))
    degree_distro(g_)
    print('degree distribution plotted')

if centrality:
    # simplify graph by removing loops and multiple edges; weights are added
    g_ = g.simplify(multiple=True, combine_edges=dict(weight='sum', date='ignore', sentiment='ignore'))
    dc(g_)
    print('degree centrality plotted')

# distribution of sentiment in e-mails
if sentiment:
    sentiment_analysis(data.copy())
    print('sentiment distribution plotted')

# if wanted, plot distribution of interaction time differences
if interaction:
    interaction_time(data.copy())
    print('Interaction time differences plotted')

# plot e-mail activity over time
if activity:
    activity_plot(data.copy())
    print('e-mail activity plotted')


########################################################################
# GRAPHS

# if wanted, draw network with communities
if draw_static:
    # simplify graph
    g_ = g.simplify(multiple=True, combine_edges=dict(weight='sum', date='ignore', sentiment='ignore'))
    g_ = g.as_undirected(combine_edges=dict(weight='sum', date='ignore'))

    # print graph information
    num_vertices_ = len(g_.vs) # number of vertices
    num_links_ = len(g_.es)

    print('graph information: {} nodes, {} links'.format(num_vertices_, num_links_))

    # detect communities
    communities = g_.community_fastgreedy(weights='weight').as_clustering()
    community_dict = dict((x,communities.membership.count(x)) for x in set(communities.membership))  # communities and number of members

    # calculate edges between communities
    # possible connections
    connections = [comb for comb in combinations(np.arange(max(communities.membership)+1), 2)]
    connections_dict = dict(zip(connections, np.zeros(len(connections))))

    for i, edge in tqdm(enumerate(g_.get_edgelist()), total=len(g_.get_edgelist()), unit='edges'):
        # get communities of edge partners
        com1 = communities.membership[edge[0]]
        com2 = communities.membership[edge[1]]
        # do not consider internal edges
        if com1 == com2:
            continue
        # lower index first for indexing
        if com1 > com2:
            conn = (com2, com1)
        else:
            conn = (com1, com2)
        # increase connection weight between according communities
        connections_dict[conn] += g_.es[i]['weight']

    # remove connections if no edge exits
    connections_dict = {x:y for x,y in connections_dict.items() if y!=0}
    # weigts ordered like edge_list
    com_weights = [value for (key, value) in sorted(connections_dict.items())]

    # only one vertex per community
    g_con = communities.cluster_graph()
    # assigning weights to edges
    g_con.es['weight'] = com_weights

    print('communities detected')

    # define visual styling
    visual_style = {}

    # define visual styling
    max_size = 5 # maximum vertex size
    min_width = 0.5 # maximum edge size

    visual_style['vertex_size'] = np.log(np.array(list(community_dict.values())))*max_size
    visual_style['layout'] = g_con.layout_fruchterman_reingold()
    visual_style['vertex_color'] = list(community_dict.keys())
    visual_style['palette'] = ig.ClusterColoringPalette(len(list(community_dict.keys())) + 3)
    visual_style['edge_width'] = np.log(np.array(g_con.es['weight']))+min_width
    visual_style['edge_color'] = 'rgba(0,0,0,0.1)'
    visual_style['mark_groups'] = False
    visual_style['margin'] = 50
    visual_style['bbox'] = (700,700)

    # plot graph grouped by communities
    ig.plot(g_con, './figures/communities_united.png', **visual_style)

    print('communities drawn')

    # circos plot
    visual_style['layout'] = g_con.layout_circle()
    ig.plot(g_con, './figures/circos_united.png', **visual_style)

    print('circos plot made')

# draw graph each month
if draw_dynamic:

    # weights
    weights = g.es['weight']

    # get layout for all times
    # define visual styling
    visual_style = {}

    # define visual styling
    max_size = 40 # maximum vertex size

    color_indices = np.arange(len(g.vs))
    visual_style['layout'] = g.layout_fruchterman_reingold()
    visual_style['vertex_color'] = list(np.arange(len(g.vs)))
    visual_style['palette'] = ig.ClusterColoringPalette(len(color_indices) + 3)
    visual_style['margin'] = (40, 120, 40, 40)
    visual_style['bbox'] = (20, 20, 980, 980)
    visual_style['edge_color'] = 'rgba(0,0,0,0.15)'

    current = start + relativedelta(months=1)
    i = 0

    while current <= end:
        # last month
        before = current - relativedelta(months=1)

        # time range
        time_range = before.strftime('%B %Y')
        print(time_range)

        # select edges within outlying time range and construct graph
        hidden_edges1 = g.es.select(date_lt=time.mktime(before.timetuple()))
        hidden_edges2 = g.es.select(date_ge=time.mktime(current.timetuple()))
        # initalize weights as if for all times
        g.es['weight'] = weights
        # change weights of hidden edges to 0
        hidden_edges1['weight'] = 0
        hidden_edges2['weight'] = 0
        # visualization
        visual_style['edge_width'] = np.array(g.es['weight'])/4
        visual_style['edge_arrow_size'] = np.array(g.es['weight'])/16
        # degree for vertex_size
        strength = g.strength(weights='weight')
        visual_style['vertex_size'] = np.array(strength)/max(strength)*max_size

        # plot graph grouped by communities
        plot = ig.Plot('./figures/time/graph-%s.png' % str(i), bbox=(1000, 1000), background="white")
        plot.add(g, **visual_style)
        # Make the plot draw itself on the Cairo surface
        plot.redraw()
        # Grab the surface, construct a drawing context and a TextDrawer
        ctx = cairo.Context(plot.surface)
        ctx.set_font_size(28)
        drawer = TextDrawer(ctx, time_range, halign=TextDrawer.CENTER)
        drawer.draw_at(0, 80, width=1000)

        # Save the plot
        plot.save()
        print('graph saved')

        # current date + 1 month
        current = current + relativedelta(months=1)
        # increase counter
        i += 1


########################################################################
# TIME ANALYSIS / DYNAMICS

# detect changes
if detect_change:
    # centrality
    measure_change(g, start, end, measure='centrality')
    # strength
    measure_change(g, start, end, measure='strength')
    # sentiment
    measure_change(g, start, end, measure='sentiment')


# analysis dynamical case
if dynamic:
    # with decay
    dynamics_analysis(g, start, end, centrality_ana=True, decay=True, limit=0.01)
    # without decay
    dynamics_analysis(g, start, end, centrality_ana=True, decay=False, limit=0.01)


########################################################################
# MAIN
if __name__ == '__main__':
    print('successful run')
