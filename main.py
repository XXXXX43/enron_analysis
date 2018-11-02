# credential for the static network analysis go to James Tollefson: https://www.kaggle.com/jamestollefson/enron-network-analysis

import numpy as np
import pandas as pd
import os
import argparse
import re
from functions.prepare import prepare_data # from folder functions import from the prepare file the function prepare_data
import matplotlib.pyplot as plt

# parser function to collect and pass on values from terminal
def parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--print", help='Should I print?', type=bool, default=False)
    parser.add_argument("--data_link", help='Link to data that has to be loaded', type=str, default='./data/emails_kaggle.csv')
    args = parser.parse_args()

    return args.data_link

# pass on parser values
data_link = parser()

# load data
pd.options.mode.chained_assignment = None
chunk = pd.read_csv(data_link, chunksize=500)
data = next(chunk)

# printing data information
#data.info()
#print(data.message[2])

data = prepare_data(data)

print(data.head())



# main function
if __name__ == '__main__':
    print('Hello World')

