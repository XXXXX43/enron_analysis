########################################################################
# IMPORT

# basic modules
import numpy as np
import pandas as pd
from datetime import datetime
# text processing
from textblob import TextBlob
from nltk.tokenize import TabTokenizer
#from textblob.sentiments import NaiveBayesAnalyzer # for sentiment analysis


########################################################################
# PROCESSING

# function to sort e-mail partners
def conv_partner(partner1, partner2):
    partners = [partner1, partner2]
    sorted_partners = np.sort(partners)
    return sorted_partners[0] + '-' + sorted_partners[1]

def process_text(message):
    # init
    sender = ''
    recipient = ''
    #recipient2 = ''
    subject = ''
    text = ''
    in_body = False
    date = ''
    date_format = "%d %b %Y %H:%M:%S" # date format
    #forwarded = False

    # split text in sentences
    sentences = message.split('\n')

    for s in sentences:

        # get sender information
        if 'From: ' in s and sender == '':
            sender = s[6:]
        # get recipient information
        if 'To: ' in s and recipient == '':
            recipient = s[4:]
        # get subject information
        if 'Subject: ' in s and subject == '':
            subject = s[9:]
            #if 'FW: ' in subject:
                #forwarded = True
        # get date/time information
        if 'Date: ' in s and date == '':
            date = s[11:-12]
            date = datetime.strptime(date, date_format)
        # check if part of email body
        if 'X-FileName: ' in s and not in_body:
            in_body = True
            continue # skip this sentence because body starts next sentence
        # if in body add sentence to text
        if in_body:
            text = text + s

    # conversation partners
    conv_p = conv_partner(sender, recipient)
    # sentiment
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    # annulate emails with wrong date annotations
    if date.year < 1980:
        date = ''

    # email has to contain information about sender, recipient and date
    if '' in [sender, recipient, date]:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        return sender, recipient, subject, text, date, conv_p, sentiment
