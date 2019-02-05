# basic modules
import numpy as np
import pandas as pd
# network analysis
import igraph as ig
# time analysis
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# external functions
from functions.plot_routines import lineplot, boxplot


def dynamics_analysis(graph, start, end, centrality_ana=True, decay=True, factor_remain=0.99, min_ver=500, limit=0.01):

    '''
    parameter:
    - graph: graph with weights and date as edge attribute
    - start (datetime): start date
    - end (datetime): end date
    - centrality: Analyze centrality? (computational intensive); bool
    - decay: data from past with decay of influence; bool
    - factor_remain: factor remaining weight per day (e.g. 0.99); int
    - min_ver: mimimum number of vertices to calculate measures; int
    '''

    # information to extract
    dates_str = []
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

    # create graphs depending on months
    current = start + relativedelta(months=1)
    while current <= end:
        # last month
        before = current - relativedelta(months=1)

        # time range
        time_range = before.strftime('%B %Y')
        print(time_range)

        # select edges within time range and construct graph
        current_edges = graph.es.select(date_lt=time.mktime(current.timetuple()))

        if decay:
            # adjust weights depending on age
            for e in current_edges:
                delta = current - datetime.fromtimestamp(e['date'])
                e['weight'] = e['weight'] * factor_remain**delta.days

            # delete edges with weight under certain limit
            current_edges = current_edges.select(weight_ge=limit)

        # create current subgraph
        current_g = graph.subgraph_edges(current_edges, delete_vertices=True)

        # at least minimum number of vertices
        ver = len(current_g.vs)

        if ver < min_ver:
            print('skip')
            # current date + 1 month
            current = current + relativedelta(months=1)
            continue

        # get measures
        num_ver.append(ver)  # number of vertices
        degrees = np.array(current_g.degree())  # degree data
        max_deg.append(np.max(degrees))  # maximum degree
        mean_deg.append(np.mean(degrees))  # mean degree
        deg.append(degrees)  # complete degree data
        if centrality_ana:
            current_cent = np.array(current_g.evcent(directed=True, weights='weight'))
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

        # store string of date
        dates_str.append(time_range)

        # current date + 1 month
        current = current + relativedelta(months=1)

    ##### PLOTTING

    # paths to save files
    if decay:
        path_to_save = './statistics/dynamics/decay/yes/'

    else:
        path_to_save = './statistics/dynamics/decay/no/'

    # size of figure
    figuresize = (20, 10)

    # plot vertices
    lineplot(figuresize, dates_str, num_ver, 'date', '#vertices', 'Dynamics of Vertices', True, path_to_save + 'vertices.png')
    # plot degree mean
    lineplot(figuresize, dates_str, mean_deg, 'date', 'degree', 'Dynamics of Degree Mean', True, path_to_save + 'degree_mean.png')

    # plot degree
    boxplot(data=deg, fig_size=figuresize, x_ticks=dates_str, x_label='date', y_label='degree', title='Dynamics of Degree', rotate=True, save_path=path_to_save + 'degree.png', log=False)

    if centrality_ana:
        # plot centrality mean
        lineplot(figuresize, dates_str, mean_cent, 'date', 'centrality', 'Dynamics of Eigenvector Centrality Mean', True, path_to_save + 'centrality_mean.png', log=True)
        boxplot(data=cent, fig_size=figuresize, x_ticks=dates_str, x_label='date', y_label='centrality', title='Dynamics of Eigenvector Centrality', rotate=True, save_path=path_to_save + 'centrality.png', log=True)

    # plot degrees of people
    plt.figure(figsize=figuresize)
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
    plt.figure(figsize=figuresize)
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
