import matplotlib.pyplot as plt
import numpy as np
import igraph as ig

# fit function
def fit_gamma(G, kmin, kmax):
    krange = np.array(range(kmin, kmax+1))
    data = np.array([k for k in G.degree() if k in krange])
    return 1 + len(data) / np.sum(np.log(data / (kmin - 0.5)))

def degree_distro(g):
    # get maximum degree
    degrees = g.degree()
    id_max = np.argmax(degrees)
    degree_max = degrees[id_max]

    # degree distribution
    degree_distro = g.degree_distribution() # degree distribution
    k_p = np.array(list(degree_distro.bins()))[:,::2].T
    k = k_p[0] # degrees
    p_k = k_p[1] / np.sum(k_p[1]) # probabilities

    # bins of logscale size for degree distribution
    bins = np.logspace(0, np.log10(degree_max), 100)
    hist, _ = np.array(np.histogram(g.degree(), bins=bins))
    hist = hist / len(g.degree())

    # fit to linear regions
    kmin = 1
    kmid = 50
    kmax = degree_max
    gamma_low = fit_gamma(g, kmin, kmid)
    gamma_high = fit_gamma(g, kmid, kmax)

    # plot
    plt.figure(figsize=(20,15))
    plt.subplot(121)
    plt.loglog(bins[:-1], hist, '.')
    plt.loglog(k[kmin:kmid], k[kmin:kmid]**(-gamma_low)/np.sum(k[kmin:kmid]**(-gamma_low)), \
            label = "$\gamma = {}$".format(round(gamma_low, 3)))
    plt.loglog(k[kmid:kmax], k[kmid:kmax]**(-gamma_high)/np.sum(k[kmid:kmax]**(-gamma_high)), \
            label = "$\gamma = {}$".format(round(gamma_high, 3)))
    plt.title("Degree distribution of Enron e-mail graph".format(bins.shape[0]))
    plt.ylabel("$p_k$")
    plt.xlabel("$k$")
    plt.legend()
    plt.savefig('./statistics/distro.png')

    # cleaning plots
    plt.gcf().clear()

    return None
