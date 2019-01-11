import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def lineplot(fig_size, x, y, x_label, y_label, title, rotate, save_path):
    '''
    parameter:
    - fig_size: size of plot (e.g. (10,10)); tuple
    - x,y: data; arrays
    - x_label and y_label: labels for axis; string
    - title: title of plot; string
    - rotate: rotate xticks; bool
    - save_path: Where has file to be saved?; string
    '''

    plt.figure(figsize=fig_size)
    sns.set_style("whitegrid")
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if rotate:
        plt.gca().set_xticklabels(x, rotation=45)

    plt.savefig(save_path)

    # cleaning plots
    plt.gcf().clear()

def boxplot(data, fig_size, x_ticks, x_label, y_label, title, rotate, save_path, log=False):

    plt.figure(figsize=fig_size)
    sns.set_style("whitegrid")
    # Create the boxplot

    ## to get fill color
    bp = plt.boxplot(data, patch_artist=True, showfliers=False)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    # if wanted set logscale
    if log:
        plt.yscale('log')

    plt.gca().set_xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # set x tick labels
    if rotate:
        plt.gca().set_xticklabels(x_ticks, rotation=45)
    else:
        plt.gca().set_xticklabels(x_ticks)

    plt.savefig(save_path)

    # cleaning plots
    plt.gcf().clear()

def violinplot(data, fig_size, x_ticks, x_label, y_label, title, rotate, save_path, log=False):

    plt.figure(figsize=fig_size)
    ax = plt.gca()
    sns.set_style("whitegrid")
    # Create the violin
    plt.violinplot(data, showextrema=False)

    ax.set_xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # if wanted set logscale
    if log:
        plt.yscale('log')
    # hide every second tick
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    # dublicate xticks
    x_ticks = [val for val in x_ticks for _ in (0, 1)]
    # set xtick labels
    if rotate:
        ax.set_xticklabels(x_ticks, rotation=45)
    else:
        ax.set_xticklabels(x_ticks)

    plt.savefig(save_path)

    # cleaning plots
    plt.gcf().clear()
