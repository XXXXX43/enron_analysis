import matplotlib.pyplot as plt
import seaborn as sns

def td(data):

    plt.figure(figsize=(20,20))
    plot = sns.distplot(data, rug=True, hist=False)
    #plt.yscale('log')

    plt.xlabel('time between interactions [s]')
    plt.ylabel('percentage')
    plt.title('Time differences interactions')
    plt.savefig('./figures/time_differences.png')

    # cleaning plots
    plt.gcf().clear()

    return None
