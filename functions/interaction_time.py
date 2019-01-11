import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def interaction_time(data):

    # group conversation partners
    conv_partners = data.groupby('conv_partner')

    # for each pair calcualte time differences between conversations
    time_dif = []
    for x in conv_partners.groups:
        conv_data = conv_partners.get_group(x)
        # add time differenc column
        conv_data = conv_data.sort_values(by=['date'])
        conv_data['time_dif'] = conv_data['date'] - conv_data['date'].shift(1) # time difference column next e-mail
        conv_data['time_dif'] = conv_data['time_dif'].apply(lambda x: x.total_seconds()) # time differences in seconds
        conv_data = conv_data.dropna() # drop NAN values
        conv_data = conv_data[conv_data['time_dif'] > 1] # time difference less then a second makes no sense
        # append time differences of this pair (time difference in hours)
        time_dif.extend(conv_data['time_dif'].values/3600)

    # bins of logscale size for degree distribution
    bins = np.logspace(0, np.log10(max(time_dif)), 100)

    #plot
    plt.xscale('symlog')
    plt.hist(time_dif, bins, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel("$t[h]$")
    plt.ylabel('Probability [%]')
    plt.title("Distribution of Interaction Time Differences of Conversation Partners")
    plt.grid(True)

    plt.savefig('./statistics/time_dif.png')

    # cleaning plots
    plt.gcf().clear()

    return None
