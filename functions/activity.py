import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# time analysis
from datetime import datetime


def activity_plot(data):

    # convert time to datetime
    data['date'] = data[['date']].apply(lambda x: datetime.fromtimestamp(x), axis=1)

    # group  time data monthly
    monthly_activity = data.groupby([pd.Grouper(key='date', freq='M'), 'sender']).count()['recipient'].reset_index()

    # plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 10))
    ax = plt.gca()  # get current axis
    sns.boxplot(x="date", y="recipient", data=monthly_activity, showfliers=False, color="salmon", saturation=.5, medianprops=dict(color='blue'))
    ax.set_xticklabels([pd.to_datetime(tm).strftime("%Y-%m") for tm in monthly_activity['date'].unique()], rotation=45) # rename xticks
    plt.ylabel('monthly e-mails')
    plt.title('E-Mail Activity')

    plt.savefig('./statistics/activity.png')

    # cleaning plots
    plt.gcf().clear()

    return None
