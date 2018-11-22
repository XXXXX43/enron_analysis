import matplotlib.pyplot as plt
import networkx as nx
import nxviz as nv

def make_circosplot(G):
    plt.figure(figsize=(20,20))
    plot = nv.CircosPlot(G)
    plot.draw()
    plt.title('Circosplot of Network')
    plt.savefig('./figures/circosplot.png')

    # claening plots
    plt.gcf().clear()

    return None
