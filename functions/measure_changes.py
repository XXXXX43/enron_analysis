import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import time
from tqdm import tqdm


def measure_change(graph, start, end, measure='centrality'):

    '''
    Parameters:
    - graph (igraph graph object): network to investigate
    - start (datetime): start date
    - end (datetime): end date
    - measure (string): Which characteristic to investigate?
    '''

    current = start + relativedelta(months=1)
    i = 0

    participants = []
    measures = []
    dates = []

    while current <= end:
        # last month
        before = current - relativedelta(months=1)

        # time range
        time_range = before.strftime('%B %Y')
        print(time_range)

        # select edges within time range and construct graph
        current_edges = graph.es.select(date_ge=time.mktime(before.timetuple()))
        current_edges = current_edges.select(date_lt=time.mktime(current.timetuple()))

        # create current subgraph
        current_g = graph.subgraph_edges(current_edges, delete_vertices=True)

        if measure == "centrality":
            # calculate centralities
            current_measures = current_g.evcent(directed=True, weights='weight')
        if measure == "strength":
            # calculate strengths
            current_measures = current_g.strength(weights='weight')
        if measure == "sentiment":
            current_measures = []
            # get median of sentiment for each vertex
            for vertex in current_g.vs:
                edges = current_g.es.select(_source_in = [vertex.index])
                current_measures.append(np.median(np.array(edges['sentiment'])))

        # store data
        participants.extend(current_g.vs['name'])
        measures.extend(current_measures)
        dates.extend([before] * len(current_measures))

        # current date + 1 month
        current = current + relativedelta(months=1)
        # increase counter
        i += 1

    # make pandas dataframe
    measures_df = pd.DataFrame(
                        {'name': participants,
                         'measure': measures,
                         'date': dates})

    # drop nan
    measures_df.dropna(subset=['measure'], inplace=True)

    difference_measures = []
    names = []
    months = []
    # group person wise
    persons = measures_df.groupby('name')

    for x in tqdm(persons.groups, unit='person'):
        personal_data = persons.get_group(x)
        # add time differenc column
        personal_data = personal_data.sort_values(by=['date'])
        personal_data['time_dif'] = personal_data['date'] - personal_data['date'].shift(1)
        # get desired time difference in months, because only monthly differences are investigated
        personal_data['desired_dif'] = personal_data['date'].apply(lambda x: x - (x - relativedelta(months=1))) # time differences in seconds
        # montly difference?
        personal_data['monthly'] = 0
        personal_data['monthly'].loc[personal_data['desired_dif'] == personal_data['time_dif']] = 1
        # realtive differences in centralities
        personal_data['measure_dif'] = personal_data['measure'] - personal_data['measure'].shift(1)
        # clean data
        personal_data = personal_data.dropna() # drop NAN values
        personal_data = personal_data[personal_data.monthly == 1]
        # store data
        difference_measures.extend(personal_data['measure_dif'].values)
        names.extend(personal_data['name'].values)
        months.extend(personal_data['date'].values)

    # make pandas dataframe
    max_abs = np.max(np.abs(difference_measures))  # to get relative differences
    difference_measures = np.array(difference_measures)/max_abs  # convert to numpy array and norm
    dif_measures_df = pd.DataFrame(
                        {'name': names,
                         'dif_measure': difference_measures,
                         'date': months})

    # getting highest changes
    top_value_upper = np.sort(difference_measures)[-1]
    top_value_lower = np.sort(difference_measures)[0]
    dif_measures_df1 = dif_measures_df[(dif_measures_df.dif_measure == top_value_upper)]
    dif_measures_df2 = dif_measures_df[(dif_measures_df.dif_measure == top_value_lower)]
    dif_measures_df_top = pd.concat([dif_measures_df1, dif_measures_df2])
    # time sort
    dif_measures_df_top.sort_values(by=['date'], inplace=True)
    # convert date to string
    dif_measures_df_top['date'] = dif_measures_df_top['date'].apply(lambda d: d.strftime('%B %Y'))
    # save as csv
    path = './statistics/dynamics/changes/' + measure
    path = path + '/'
    path = path + measure

    dif_measures_df_top.to_csv(path + '_changes_top.csv', encoding='utf-8', index=False)

    # plot
    # histogram
    n, bins, patches = plt.hist(difference_measures, 100, density=True, facecolor='salmon', alpha=0.5, label="data")
    # best fit of data (exclude 0 bin)
    (mu, sigma) = norm.fit(difference_measures[np.nonzero(difference_measures)])
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'b--', linewidth=2, label=r'$\mathrm{fit:}\ \mu=%.2f,\ \sigma=%.2f$' %(mu, sigma), alpha=0.5)
    # comment that 0 bin excluded
    plt.plot([], [], ' ', label="(0 bin excluded)")
    # style plot
    #plt.yscale('log')
    plt.xlim((-0.3, 0.3))
    plt.xlabel('difference')
    plt.ylabel('probability [%]')
    plt.title("Monthly Differences: " + measure)
    plt.legend()
    ax = plt.gca()  # get current axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.grid(True)
    # save
    plt.savefig(path + '_change.png')
    # cleaning plots
    plt.gcf().clear()

    # store changes that are larger then one sigma
    dif_measures_df_sigma = dif_measures_df[(dif_measures_df.dif_measure > 3*sigma) | (dif_measures_df.dif_measure < (-3)*sigma)]
    # time sort
    dif_measures_df_sigma.sort_values(by=['date'], inplace=True)
    # convert date to string
    dif_measures_df_sigma['date'] = dif_measures_df_sigma['date'].apply(lambda d: d.strftime('%B %Y'))
    # save as csv
    dif_measures_df_sigma.to_csv(path + '_changes_sigma.csv', encoding='utf-8', index=False)
