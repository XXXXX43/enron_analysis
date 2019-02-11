import igraph as ig
import numpy as np
import matplotlib.pyplot as plt


def degree_corr(g):

    # calculate correlation matrix
    n = g.vcount()
    degrees = np.array(g.degree(), dtype=int)
    degree_max = g.maxdegree()
    e = np.zeros((degree_max+1, degree_max+1))

    for i in range(n):
        # degree of node i
        k_i = degrees[i]
        # degrees of i's neighbors
        k_js = degrees[g.neighbors(i)]
        # store in e
        for k_j in k_js:
            e[k_i, k_j] += 1.

    # normalize
    e = e / (2.*g.ecount())

    # plotting
    plt.figure(figsize=(10, 10))
    plt.title("Degree Correlation Matrix")
    plt.xlabel("$k_{1}$")
    plt.ylabel("$k_{2}$")
    # plot matrix
    plt.imshow(e[1:25, 1:25], cmap=plt.cm.get_cmap('Reds'))
    # set colorbar
    plt.colorbar(fraction=0.046, pad=0.04)
    # invert y axis
    plt.gca().invert_yaxis()
    plt.savefig('./statistics/correlations.png')

    # cleaning plots
    plt.gcf().clear()
