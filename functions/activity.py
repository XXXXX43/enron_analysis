import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# time analysis
from datetime import datetime


def activity_plot(data):

    # group  time data monthly
    monthly_activity = data.groupby([pd.Grouper(key='date', freq='M'), 'sender']).count()['recipient'].reset_index()

    # plot
    plt.figure(figsize=(25, 25))
    ax = plt.gca() # get current axis
    sns.boxplot(x="date", y="recipient", data=monthly_activity, showfliers=False)
    ax.set_xticklabels([pd.to_datetime(tm).strftime("%Y-%m") for tm in monthly_activity['date'].unique()], rotation=45) # rename xticks
    plt.ylabel('Written E-Mails/Month')
    plt.title('E-Mail Activity over Time')

    plt.savefig('./statistics/dynamics/activity.png')

    # cleaning plots
    plt.gcf().clear()

    return None
