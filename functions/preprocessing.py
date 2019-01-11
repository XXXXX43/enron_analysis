# credential go to James Tollefson: https://www.kaggle.com/jamestollefson/enron-network-analysis
########################################################################

import numpy as np
import pandas as pd
import re
from datetime import datetime
# text processing
from textblob import TextBlob, Blobber
from nltk.tokenize import TabTokenizer
from textblob.sentiments import NaiveBayesAnalyzer

tb = Blobber(analyzer=NaiveBayesAnalyzer())

# defining helper functions
def get_text(Series, row_num_slicer):
    """returns a Series with text sliced from a list split from each message. Row_num_slicer
    tells function where to slice split text to find only the body of the message."""
    result = pd.Series(index=Series.index)
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        del message_words[:row_num_slicer]
        message_words = ' '.join(message_words)
        result.iloc[row] = message_words

    return result


def get_row(Series, row_num):
    """returns a single row split out from each message. Row_num is the index of the specific
    row that you want the function to return."""
    result = pd.Series(index=Series.index)
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        message_words = message_words[row_num]
        result.iloc[row] = message_words
    return result


def get_address(df, Series, num_cols=1):
    """returns a specified email address from each row in a Series"""
    # standard email format
    eformat = re.compile('[\w\.-]+@[\w\.-]+\.\w+')
    result1 = pd.Series(index=df.index)
    result2 = pd.Series(index=df.index)
    result3 = pd.Series(index=df.index)
    result4 = pd.Series(index=df.index)
    result5 = pd.Series(index=df.index)
    for i, row in enumerate(Series):
        correspondents = re.findall(eformat, row)
        try:
            result1[i] = correspondents[0]
        except:
            print(correspondents, row)
        if num_cols >= 1 and len(correspondents) >= 2:
            result2[i] = correspondents[1]
        if num_cols >= 2 and len(correspondents) >= 3:
            result3[i] = correspondents[2]
        if num_cols >= 3 and len(correspondents) >= 4:
            result4[i] = correspondents[3]
        if num_cols >= 4 and len(correspondents) >= 5:
            result5[i] = correspondents[4]
    if num_cols == 1:
        return result1
    else:
        return result1, result2, result3, result4, result5


def standard_format(df, Series, string, slicer):
    """Drops rows containing messages without some specified value in the expected locations.
    Returns original dataframe without these values. Don't forget to reindex after doing this!!!"""
    indices = []
    for index, message in enumerate(Series):
        message_words = message.split('\n')
        if string not in message_words[slicer]:
            indices.append(index)
    df = df.drop(df.index[indices])
    return df


def prepare_data(data):

    # getting rid of data that misses crucial information
    x = len(data.index)
    headers = ['Message-ID: ', 'Date: ', 'From: ', 'To: ', 'Subject: ']
    for i, v in enumerate(headers):
        data = standard_format(data, data.message, v, i)
    data = data.reset_index()
    print("Got rid of {} useless emails! That's {}% of the total number of messages in this dataset.".format(x - len(data.index), np.round(((x - len(data.index)) / x) * 100, decimals=2)))

    # get date information
    data['date'] = get_row(data.message, 1)
    data.date = data.date.str.replace('Date: ', '')
    data.date = pd.to_datetime(data.date)
    # get information of sender
    data['senders'] = get_row(data.message, 2)
    data['sender'] = get_address(data, data.senders)
    del data['senders']
    data.dropna(subset=['sender'], inplace=True)
    # recipients
    data['recipients'] = get_row(data.message, 3)
    data['recipient'], data['recipient2'], data['recipient3'], data['recipient4'], data['recipient5'] = get_address(data, data.recipients, num_cols=5)
    del data['recipients']
    data.dropna(subset=['recipient'], inplace=True)
    # sentiment using probability for positiv sentiment
    data['text'] = get_text(data.message, 15)
    data['sentiment'] = data['text'].apply(lambda x: tb(x).sentiment[1])
    # delete unneccessary columns
    del data['file']
    del data['message']
    del data['text']

    return data
