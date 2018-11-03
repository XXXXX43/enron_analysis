import matplotlib.pyplot as plt
import networkx as nx
import nxviz as nv

def make_arcplot(G):
    plt.figure(figsize=(20,20))
    plot = nv.ArcPlot(G)
    plot.draw()
    plt.title('Arcplot of Network')
    plt.savefig('./figures/arcplot.png')

    # cleaning plots
    plt.gcf().clear()

    return None
