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

    # the histogram of the data
    n, bins, patches = plt.hist(data['sentiment'].values, 60, density=True, facecolor='salmon', alpha=0.5)

    #plot
    plt.yscale('log')
    plt.xlabel('certainty classified positive')
    plt.ylabel('probability')
    plt.title("Classification Sentiment")
    plt.grid(True)

    plt.savefig('./statistics/sentiment_distro.png')

    # cleaning plots
    plt.gcf().clear()

    ##### DYNAMICS

    # convert date
    data['date'] = data[['date']].apply(lambda x: datetime.fromtimestamp(x), axis=1)

    # time range with enough data per month
    data = data.where((data['date'] >= datetime(1998, 11, 1)) & (data['date'] < datetime(2002, 8, 1)))

    # group  time data monthly
    monthly_sentiment = data.groupby([pd.Grouper(key='date', freq='M'), 'sender']).mean()['sentiment'].reset_index()

    # plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 10))
    ax = plt.gca() # get current axis
    sns.boxplot(x="date", y="sentiment", data=monthly_sentiment, showfliers=False, color="salmon", saturation=.5, medianprops=dict(color='blue'))
    ax.set_xticklabels([pd.to_datetime(tm).strftime("%Y-%m") for tm in monthly_sentiment['date'].unique()], rotation=45) # rename xticks
    plt.ylabel('sentiment')
    plt.title('Time-dependent Sentiment')

    plt.savefig('./statistics/sentiment.png')

    # cleaning plots
    plt.gcf().clear()


    return None
