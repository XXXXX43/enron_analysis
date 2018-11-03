import matplotlib.pyplot as plt
import networkx as nx
import nxviz as nv

def draw(G):
    plt.figure(figsize=(20,20))
    pos = nx.spring_layout(G, k=.1)
    nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=True, edge_color='blue')
    plt.title('Drawing of Network')
    plt.savefig('./figures/network_drawing.png')

    # claening plots
    plt.gcf().clear()

    return None
