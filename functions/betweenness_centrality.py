import matplotlib.pyplot as plt
import networkx as nx
import nxviz as nv
import pandas as pd
import seaborn as sns

def bc(G):
    between = nx.betweenness_centrality(G)
    name = []
    betweenness = []

    for key, value in between.items():
        name.append(key)
        betweenness.append(value)

    bet = pd.DataFrame()
    bet['name'] = name
    bet['betweenness'] = betweenness
    bet = bet.sort_values(by='betweenness', ascending=False)


    plt.figure(figsize=(25, 25))
    _ = sns.barplot(x='betweenness', y='name', data=bet[:15], orient='h')
    _ = plt.xlabel('Degree Betweenness Centrality')
    _ = plt.ylabel('Correspondent')
    _ = plt.title('Top 15 Betweenness Centrality Scores in Enron Email Network')

    plt.savefig('./figures/betweenness_centrality.png')

    # cleaning plots
    plt.gcf().clear()

    return None
