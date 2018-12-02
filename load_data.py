import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

price_path = 'data/2012_election_prices.csv'
polls_path = 'data/2012_election_polls.csv'

def load(price_path, polls_path):
    prices = pd.read_csv(price_path, sep='\t', dtype={u'Date': str, u'Open': float, u'High': float, u'Low Close': float, u'Volume': int}, index_col=False)
    polls = pd.read_csv(polls_path, sep='\t', dtype={u'Poll': str, u'Date': str, u'Sample': str, u'MoE': str, u'Obama (D)': float, u'Romney (R)': float, u'Spread': str}, index_col=False)

    # fix dates for prices
    prices['Date'] = prices['Date'].apply(lambda x: pd.Timestamp(x).date())
    prices = prices.loc[prices['Date'] >= pd.Timestamp('2011-01-06').date()]

    # fix sample numbers for polls
    def parsesample(samplestr):
        return int(samplestr.split()[0]) if samplestr.split()[0] != 'RV' else None
    polls['Sample'] = polls['Sample'].apply(parsesample)

    # fix dates for polls
    dates = []
    def parsedate(datestr, year):
        return pd.Timestamp(datestr.split('-')[1] + '/' + str(year))
    for i, date in polls['Date'].iteritems():
        if i < 202:
            dates.append(parsedate(date, 12))
        else:
            dates.append(parsedate(date, 11))
    polls['Date'] = dates
    polls.sort_values(by=['Date'], ascending=True, inplace=True)
    groupedpolls = polls.groupby(polls['Date'].dt.date).mean().rename(columns={'size':'mean'})
    groupedpolls.reset_index(level=0, inplace=True)
    merge = pd.merge(prices, groupedpolls, how='left', on=['Date'])

    merge = merge.fillna(method='ffill')
    return merge
