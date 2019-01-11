# credential go to James Tollefson: https://www.kaggle.com/jamestollefson/enron-network-analysis
########################################################################

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def dc(g):
    # getting values
    centrality = g.betweenness(directed=True)
    name = g.vs['name']
    # making pandas dataframe
    cent = pd.DataFrame()
    cent['name'] = name
    cent['centrality'] = centrality
    cent = cent.sort_values(by='centrality', ascending=False)
    # plot
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(40, 15))
    _ = sns.barplot(x='centrality', y='name', data=cent[:15], orient='h')
    _ = plt.xlabel('Degree Centrality')
    _ = plt.ylabel('Person')
    _ = plt.title('Top 15 Degree Centrality Scores in Enron Email Network')

    plt.savefig('./statistics/degree_centrality.png')

    # cleaning plots
    plt.gcf().clear()

    return None
