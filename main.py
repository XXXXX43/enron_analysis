import numpy as np
import pandas as pd
import os
import argparse
#from functions.VAE_pixel import train # from folder functions import form the VAE_pixel file the function train


print('ok')

# parser function to collect and pass on values from terminal
def parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--print", help='Should I print?', type=bool, default=False)
    parser.add_argument("--data_link", help='Link to data that has to be loaded', type=str, default='./data/emails_kaggle.csv')
    args = parser.parse_args()

    return args.data_link

# pass on parser values
data_link = parser()

print(data_link)

# load data
pd.options.mode.chained_assignment = None
chunk = pd.read_csv(data_link, chunksize=500)
data = next(chunk)

data.info()
print(data.message[2])

# main function
if __name__ == '__main__':
    print('Hello World')

