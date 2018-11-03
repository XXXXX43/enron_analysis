import matplotlib.pyplot as plt
import networkx as nx
import nxviz as nv
import pandas as pd
import seaborn as sns

def dc(G):
    cent = nx.degree_centrality(G)
    name = []
    centrality = []

    for key, value in cent.items():
        name.append(key)
        centrality.append(value)

    cent = pd.DataFrame()
    cent['name'] = name
    cent['centrality'] = centrality
    cent = cent.sort_values(by='centrality', ascending=False)

    plt.figure(figsize=(25, 25))
    _ = sns.barplot(x='centrality', y='name', data=cent[:15], orient='h')
    _ = plt.xlabel('Degree Centrality')
    _ = plt.ylabel('Correspondent')
    _ = plt.title('Top 15 Degree Centrality Scores in Enron Email Network')

    plt.savefig('./figures/degree_centrality.png')

    # cleaning plots
    plt.gcf().clear()

    return None
