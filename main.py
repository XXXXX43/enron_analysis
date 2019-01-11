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
# for parallelization
import concurrent.futures
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
# functions loaded form other files
#from functions.processing import process_text # from folder functions import from the prepare file the function prepare_data
from functions.degree_distro import degree_distro
from functions.degree_centrality import dc
from functions.preprocessing import prepare_data
from functions.sentiment_analysis import sentiment_analysis
from functions.interaction_time import interaction_time
from functions.activity import activity_plot
from functions.plot_routines import lineplot, boxplot, violinplot


########################################################################
# PARSING

# parser function to collect and pass on values from terminal
def parser():
    parser = argparse.ArgumentParser(description='Specify analysing process')
    parser.add_argument("--data_link", help='Link to data that has to be loaded', \
        type=str, default='./data/prepared/total.txt')
    parser.add_argument("--cols", help='Which columns of data to use: sender, recipient, recipient2, recipient3, subject, text, date, conv_partner, sentiment?', \
        type=str, default='sender recipient recipient2 recipient3 recipient4 recipient5 date sentiment')
    parser.add_argument("--stat", help='Calculate statistics', \
        action='store_true') # default is false
    parser.add_argument("--not_load", help='Prepare data from scratch?', \
        action='store_false') # default is true
    parser.add_argument("--draw_com", help='Draw network with communities', action='store_true') # default is false
    parser.add_argument("--distro", help='Degree distribution', action='store_true') # default is false
    parser.add_argument("--circos_plot", help='Make circos plot', action='store_true') # default is false
    parser.add_argument("--centrality", help='Show degree centrality', action='store_true') # default is false
    parser.add_argument("--sentiment", help='Plot sentiment distribution', action='store_true') # default is false
    parser.add_argument("--interaction", help='Plot interaction time differences', action='store_true') # default is false
    parser.add_argument("--activity", help='Plot e-mail activity over time', action='store_true') # default is false

    args = parser.parse_args()

    data_link = args.data_link
    not_load = args.not_load
    draw_com = args.draw_com
    cols = args.cols.split()
    circos_plot = args.circos_plot
    stat = args.stat
    distro = args.distro
    centrality = args.centrality
    sentiment = args.sentiment
    interaction = args.interaction
    activity = args.activity

    return data_link, not_load, draw_com, cols, circos_plot, stat, distro, centrality, sentiment, interaction, activity

# pass on parser values
data_link, not_load, draw_com, cols, circos_plot, stat, distro, centrality, sentiment, interaction, activity = parser()


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

# focus on data of certain time range (1998-10-1 to 2002-07-31)
date_format = "%Y-%m-%d %H:%M:%S" # date format
data['date'] = pd.to_datetime(data['date'], format=date_format)  # convert to type datetime
data = data.where((data['date'] >= datetime(1998, 10, 1)) & (data['date'] < datetime(2002, 8, 1)))
#data = data.dropna(axis=0, how='any')

########################################################################
# MAKING GRAPH

# maximum of 5 recipients in dataset
# recipient
data1 = data.copy()
data1.drop(['recipient2', 'recipient3', 'recipient4', 'recipient5'], axis=1, inplace=True)
# recipient2
data2 = data.copy()
data2.drop(['recipient', 'recipient3', 'recipient4', 'recipient5'], axis=1, inplace=True)
data2.dropna(subset=['recipient2'], inplace=True)
data2.rename(index=str, columns={'recipient2':'recipient'}, inplace=True)
# recipient3
data3 = data.copy()
data3.drop(['recipient', 'recipient2', 'recipient4', 'recipient5'], axis=1, inplace=True)
data3.dropna(subset=['recipient3'], inplace=True)
data3.rename(index=str, columns={'recipient3':'recipient'}, inplace=True)
# recipient4
data4 = data.copy()
data4.drop(['recipient', 'recipient2', 'recipient3', 'recipient5'], axis=1, inplace=True)
data4.dropna(subset=['recipient4'], inplace=True)
data4.rename(index=str, columns={'recipient4':'recipient'}, inplace=True)
# recipient5
data5 = data.copy()
data5.drop(['recipient', 'recipient2', 'recipient3', 'recipient4'], axis=1, inplace=True)
data5.dropna(subset=['recipient5'], inplace=True)
data5.rename(index=str, columns={'recipient5':'recipient'}, inplace=True)

# join data
frames = [data1, data2, data3, data4]
data = pd.concat(frames, axis=0, join='outer', ignore_index=True)

# create graph of data
g = ig.Graph.TupleList(data[['sender', 'recipient']].itertuples(index=False), directed=True, weights=False)
# simplify graph by removing loops and multiple edges
g = g.simplify(multiple=True, combine_edges=None)

num_vertices = len(g.vs) # number of vertices

print('graph made')

# continue working with largest connected component
g = g.clusters().giant()

########################################################################
# STATISTICS

if stat:
    # calculate and print statistics for each column
    for col in data.columns:
        data[col].describe().to_csv("./statistics/{}.csv".format(col))

if distro:
    degree_distro(g)

    print('degree distribution plotted')

if centrality:
    dc(g)

    print('degree centrality plotted')

########################################################################
# PLOTTING

# if wanted, draw network with communities
if draw_com:

    # detect communities
    communities = g.community_edge_betweenness().as_clustering()
    community_dict = dict((x,communities.membership.count(x)) for x in set(communities.membership))  # communities and number of members

    # calculate edges between communities
    # possible connections
    connections = [comb for comb in combinations(np.arange(max(communities.membership)+1), 2)]
    connections_dict = dict(zip(connections, np.zeros(len(connections))))

    for edge in tqdm(g.get_edgelist(), total=len(g.get_edgelist()), unit='edges'):
        # get communities of edge partners
        com1 = communities.membership[edge[0]]
        com2 = communities.membership[edge[1]]
        # do not consider internal edged
        if com1 == com2:
            continue
        # lower index first for indexing
        if com1 > com2:
            conn = (com2, com1)
        else:
            conn = (com1, com2)
        # increase connection weight between according communities
        connections_dict[conn] += 1

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

    visual_style['layout'] = g.layout_fruchterman_reingold()  # Fruchterman-Reingold force-directed algorithm
    visual_style['vertex_size'] = 10
    visual_style['margin'] = 1000
    visual_style['bbox'] = (10000,10000)
    visual_style['vertex_label'] = None
    visual_style['mark_groups'] = True

    # plot whole graph
    ig.plot(communities, './figures/communities.png', **visual_style)

    # define visual styling
    max_size = 20 # maximum vertex size
    max_width = 4 # maximum edge size

    visual_style['vertex_size'] = np.around(np.log(np.array(list(community_dict.values())))*max_size)
    visual_style['layout'] = g_con.layout_fruchterman_reingold()
    visual_style['vertex_color'] = list(community_dict.keys())
    visual_style['palette'] = ig.ClusterColoringPalette(len(list(community_dict.keys())) + 3)
    visual_style['edge_width'] = np.around(np.log(np.array(g_con.es['weight']))*max_width)+1
    visual_style['mark_groups'] = False

    # plot graph grouped by communities
    ig.plot(g_con, './figures/communities_united.png', **visual_style)

    print('communities drawn')

if circos_plot:

    # define visual styling
    visual_style = {}

    max_size = 20  # maximum vertex size

    visual_style['vertex_size'] = np.around(np.log(np.array(list(community_dict.values())))*max_size)
    visual_style['layout'] = g_con.layout_circle()
    visual_style['vertex_color'] = list(community_dict.keys())
    visual_style['palette'] = ig.ClusterColoringPalette(len(list(community_dict.keys())) + 3)
    visual_style['margin'] = 1000
    visual_style['bbox'] = (10000,10000)

    # plot
    ig.plot(g_con, './figures/circos.png', **visual_style)

    print('circos plot made')


########################################################################
# TIME ANALYSIS / DYNAMICS

# if wanted, plot distribution of interaction time differences
if interaction:
    interaction_time(data)
    print('Interaction time differences plotted')

# plot e-mail activity over time
if activity:
    activity_plot(data)
    print('e-mail activity plotted')


def calc_weights(data, curr, exp):

    '''
    parameter:
    - data: pandas dataframe of data, has to have column date
    - curr: current date; datetime object
    - exp: faktor reducing weight per month; int
    '''

    data['weight'] = data['date'].apply(lambda x: 1/(exp**np.abs(relativedelta(x, curr).months)))

    return data

def dynamics_analysis(data, centrality_ana=True, add_up=True, gamma=2, min_ver=500):

    '''
    parameter:
    - data: pandas dataframe of data
    - centrality: Analyze centrality? (computational intensive); bool
    - add_up: add up data from past without decay of influence; bool
    - gamma: faktor reducing weight per month; int
    - min_ver: mimimum number of vertices to calculate measures; int
    '''

    # information to extract
    dates_str = []
    #dates_const = []
    num_ver = []  # number of vertices
    mean_deg = []  # mean degrees
    max_deg = []  # max degree
    deg = [] # complete degree data
    mean_cent = []  # mean centrality
    cent = [] # complete centrality data
    # store degrees of some person
    mark_taylor_deg = []
    vince_kaminski_deg = []
    tana_jones_deg = []
    klay_deg = []
    # store centralities of some person
    mark_taylor_cent = []
    vince_kaminski_cent = []
    tana_jones_cent = []
    klay_cent = []

    # iterate over months in timerange
    start = datetime(1998, 1, 1)
    end = datetime(2003, 1, 1)
    #end = datetime(2000, 1, 1)
    current = start + relativedelta(months=1)

    while current <= end:
        # current date as string
        current_string = current.strftime('%d/%m/%Y')
        print(current_string)

        # data up to current month
        current_data = data.where(data['date'] < current)
        # drop nan
        current_data = current_data.dropna(axis=0, how='any')

        if not add_up:
            # calcualte weights
            current_data = calc_weights(current_data, current, exp=gamma)

        # create graph of data
        current_g = ig.Graph.TupleList(current_data[['sender', 'recipient']].itertuples(index=False), weights=False)
        # simplify graph
        if add_up:
            # simplify graph by removing loops and multiple edges
            current_g = current_g.simplify(multiple=True, loops=True, combine_edges=None)
        else:
            # add weights
            current_g.es['weight'] = current_data['weight'].values
            # simplify graph by removing loops and multiple edges, taking latest edge if multiple exist
            current_g = current_g.simplify(multiple=True, loops=True, combine_edges='max')

        # continue working with largest connected component
        #current_g = current_g.clusters().giant()

        if not add_up:
            # only take the edges that are not older then 3 months
            limit = 1/(gamma**3)
            seq = current_g.es.select(weight_gt=limit)
            current_g = current_g.subgraph_edges(seq)

        # at least 50 vertices
        ver = len(current_g.vs)

        if ver < min_ver:
            print('skip')
            # current date + 1 month
            current = current + relativedelta(months=1)
            continue

        # get measures
        num_ver.append(ver)  # number of vertices
        degrees = np.array(current_g.degree()) # degree data
        max_deg.append(np.max(degrees))  # maximum degree
        mean_deg.append(np.mean(degrees))  # mean degree
        deg.append(degrees) # complete degree data
        if centrality_ana:
            # do not use weights
            if add_up:
                current_cent = np.array(current_g.evcent(directed=False, scale=True))
                mean_cent.append(np.mean(current_cent))  # mean centrality
                cent.append(current_cent)  # complete centrality
            # do use weights
            else:
                current_cent = np.array(current_g.evcent(directed=False, scale=True, weights=current_g.es['weight']))
                mean_cent.append(np.mean(current_cent))  # mean centrality
                cent.append(current_cent)  # complete centrality

        vertices = current_g.vs['name']
        # personal data
        taylor = 'mark.taylor@enron.com'
        try:
            taylor_id = vertices.index(taylor)
            deg_taylor = degrees[taylor_id]
            cent_taylor = current_cent[taylor_id]
        except:
            deg_taylor = 0
            cent_taylor = 0

        kaminski = 'vince.kaminski@enron.com'
        try:
            kaminski_id = vertices.index(kaminski)
            deg_kaminski = degrees[kaminski_id]
            cent_kaminski = current_cent[kaminski_id]
        except:
            deg_kaminski = 0
            cent_kaminski = 0

        jones = 'tana.jones@enron.com'
        try:
            jones_id = vertices.index(jones)
            deg_jones = degrees[jones_id]
            cent_jones = current_cent[jones_id]
        except:
            deg_jones = 0
            cent_jones = 0

        klay = 'klay@enron.com'
        try:
            klay_id = vertices.index(klay)
            deg_klay = degrees[klay_id]
            cent_klay = current_cent[klay_id]
        except:
            deg_klay = 0
            cent_klay = 0

        # centrality data of certain persons
        #cent_taylor = taylor.evcent(directed=False, scale=True, weights=current_g.es['weight'])

        # store personal data
        # degree
        mark_taylor_deg.append(deg_taylor)
        vince_kaminski_deg.append(deg_kaminski)
        tana_jones_deg.append(deg_jones)
        klay_deg.append(deg_klay)
        # centrality
        mark_taylor_cent.append(cent_taylor)
        vince_kaminski_cent.append(cent_kaminski)
        tana_jones_cent.append(cent_jones)
        klay_cent.append(cent_klay)

        '''
        # check point where graph has fixed number of vertices
        if current != start:
            if len(current_g.vs) == num_ver[-2]:
                # add information
                cent.append(current_g.betweenness(directed=False)) # betweeness centrality
                deg.append(current_g.degree()) # degrees
                print(current_g.betweenness(directed=False)[-20:-10])
                #dates = [current_string] * num_ver[-1]
                #dates_const.extend(dates)
                #if len(deg) > 1:
                    #dif = np.array(deg[-1]) - np.array(deg[-2])
        '''
        # store string of date
        dates_str.append(current_string)

        # current date + 1 month
        current = current + relativedelta(months=1)

    ##### PLOTTING

    # paths to save files
    if add_up:
        path_to_save = './statistics/dynamics/'

    else:
        path_to_save = './statistics/dynamics/no_add_up/'
    '''
    # plot vertices
    lineplot((25, 25), dates_str, num_ver, 'date', '#vertices', 'Dynamics of Vertices', True, path_to_save + 'vertices.png')
    # plot degree mean
    lineplot((25, 25), dates_str, mean_deg, 'date', 'degree', 'Dynamics of Degree Mean', True, path_to_save + 'degree_mean.png')
    '''
    # plot degree
    boxplot(data=deg, fig_size=(25,25), x_ticks=dates_str, x_label='date', y_label='degree', title='Dynamics of Degree', rotate=True, save_path=path_to_save + 'degree.png', log=False)
    '''
    if centrality_ana:
        # plot centrality mean
        lineplot((25, 25), dates_str, mean_cent, 'date', 'centrality', 'Dynamics of Eigenvector Centrality Mean', True, path_to_save + 'centrality_mean.png')
        boxplot(data=cent, fig_size=(25,25), x_ticks=dates_str, x_label='date', y_label='centrality', title='Dynamics of Eigenvector Centrality', rotate=True, save_path=path_to_save + 'centrality.png')

    # plot degrees of people
    plt.figure(figsize=(25, 25))
    sns.set_style("whitegrid")
    plt.plot(dates_str, max_deg, label='maximum degree')
    plt.plot(dates_str, mark_taylor_deg, label='Mark Taylor')
    plt.plot(dates_str, vince_kaminski_deg, label='Vince Kaminski')
    plt.plot(dates_str, tana_jones_deg, label='Tanja Jones')
    plt.plot(dates_str, klay_deg, label='Kenneth Lay')
    plt.xlabel('date')
    plt.ylabel('degree')
    plt.title('Dynamics of Personal Degrees')
    plt.gca().set_xticklabels(dates_str, rotation=45)
    plt.legend(loc='best')

    plt.savefig(path_to_save + 'degree_person.png')

    # cleaning plots
    plt.gcf().clear()

    # plot centrality of people
    plt.figure(figsize=(25, 25))
    sns.set_style("whitegrid")
    plt.plot(dates_str, mark_taylor_cent, label='Mark Taylor')
    plt.plot(dates_str, vince_kaminski_cent, label='Vince Kaminski')
    plt.plot(dates_str, tana_jones_cent, label='Tanja Jones')
    plt.plot(dates_str, klay_cent, label='Kenneth Lay')
    plt.xlabel('date')
    plt.ylabel('centrality')
    plt.title('Dynamics of Personal Centrality')
    plt.gca().set_xticklabels(dates_str, rotation=45)
    plt.legend(loc='best')

    plt.savefig(path_to_save + 'centrality_person.png')

    # cleaning plots
    plt.gcf().clear()
    '''

#dynamics_analysis(data, centrality_ana=False, add_up=False)

########################################################################
# SENTIMENT ANALYSIS

# if wanted, plot distribution of sentiment in e-mails
if sentiment:
    sentiment_analysis(data)
    print('Sentiment anaysis made')

########################################################################
# MAIN
if __name__ == '__main__':
    print('successful run')
