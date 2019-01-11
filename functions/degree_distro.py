import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def fit_gamma(G, kmin, kmax, in_degree=True):
    krange = np.array(range(kmin, kmax+1))
    if in_degree:
        data = np.array([k for k in G.indegree() if k in krange])
    else:
        data = np.array([k for k in G.outdegree() if k in krange])
    return 1 + len(data) / np.sum(np.log(data / (kmin - 0.5)))


def power_law_dist(G, kmin, kmax, gamma):
    assert kmin >= 1
    assert kmax < G.vcount()

    p = np.zeros((G.vcount(),))
    for i in range(kmin, kmax+1):
        p[i] = i**(-gamma)
    p /= np.sum(p)
    return p


def degree_distro(g):
    # out degree
    bins_out = np.logspace(0, 2.8, 100)
    hist_out, _ = np.histogram(g.outdegree(), bins=bins_out)
    hist_out = hist_out / g.vcount()
    # Fit for data with degree in range [1, 800]
    gamma_out = fit_gamma(g, 1, 800, in_degree=False)
    P_out = power_law_dist(g, 1, g.vcount() - 1, gamma_out)

    # in degree
    bins_in = np.logspace(0, 2.8, 100)
    hist_in, _ = np.histogram(g.indegree(), bins=bins_in)
    hist_in = hist_in / g.vcount()
    # Fit for data with degree in range [1, 800]
    gamma_in = fit_gamma(g, 1, 800, in_degree=True)
    P_in = power_law_dist(g, 1, g.vcount() - 1, gamma_in)

    # plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 5))
    plt.loglog(bins_out[:-1], hist_out, 'b.')
    plt.loglog(range(0, g.vcount()), P_out, 'b-', label="$\gamma_o = {}$".format(round(gamma_out, 2)))
    plt.loglog(bins_in[:-1], hist_in, 'r.')
    plt.loglog(range(0, g.vcount()), P_in, 'r-', label="$\gamma_i = {}$".format(round(gamma_in, 2)))
    plt.ylabel("$p_k$")
    plt.xlabel("$k$")
    plt.xlim(1, 10**2.8)
    plt.ylim(10**-4.25, 1)
    plt.title("Degree Distribution of Enron Email Graph")
    plt.legend()
    plt.savefig('./statistics/distro.png')

    # cleaning plots
    plt.gcf().clear()

    return None
