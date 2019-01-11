import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import pandas as pd
import seaborn as sns
# time analysis
from datetime import datetime


def sentiment_analysis(data):

    ##### STATIC

    # best fit of data; excluding zero sentiment to not disturb fit
    (mu, sigma) = norm.fit(data['sentiment'].iloc[data['sentiment'].nonzero()[0]].values)

    # the histogram of the data
    n, bins, patches = plt.hist(data['sentiment'].values, 60, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)

    #plot
    plt.xlabel('Sentiment')
    plt.ylabel('Probability [%]')
    plt.text(0.2, 5, "$\mu = {0:3.2f}, \sigma = {1:3.2f}$ (0 bin excluded)".format(mu, sigma), style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.title("Distribution of Sentiment in E-Mails")
    plt.grid(True)

    plt.savefig('./statistics/sentiment_distro.png')

    # cleaning plots
    plt.gcf().clear()

    ##### DYNAMICS

    # group  time data monthly
    monthly_sentiment = data.groupby([pd.Grouper(key='date', freq='M'), 'sender']).mean()['sentiment'].reset_index()

    # plot
    plt.figure(figsize=(25, 25))
    ax = plt.gca() # get current axis
    sns.boxplot(x="date", y="sentiment", data=monthly_sentiment, showfliers=False)
    ax.set_xticklabels([pd.to_datetime(tm).strftime("%Y-%m") for tm in monthly_sentiment['date'].unique()], rotation=45) # rename xticks
    plt.ylabel('sentiment')
    plt.title('Sentiment over Time')

    plt.savefig('./statistics/dynamics/sentiment.png')

    # cleaning plots
    plt.gcf().clear()


    return None
