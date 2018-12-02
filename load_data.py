import pandas as pd
import numpy as np

price_path = 'data/2012_election_prices.csv'
polls_path = 'data/2012_election_polls.csv'

def load_data(price_path, polls_path):
    prices = pd.read_csv(price_path, sep='\t', dtype={u'Date': str, u'Open': float, u'High': float, u'Low Close': float, u'Volume': int}, index_col=False)
    polls = pd.read_csv(polls_path, sep='\t', dtype={u'Poll': str, u'Date': str, u'Sample': str, u'MoE': str, u'Obama (D)': float, u'Romney (R)': float, u'Spread': str}, index_col=False)


    # print(prices['Date'].apply(pd.Timestamp))
    # prices['Date'] = prices['Date'].apply(pd.Timestamp)
    # print(prices)


    # def parsedate(datestr, year):
    #     return datestr.split('-')[1] + '/' + str(year)

    # np.vectorize(parsedate)(polls['Date'].iloc[:202], 12)
    # np.vectorize(parsedate)(polls['Date'].iloc[202:], 11)
    # print()
    # for i, date in prices['Date'].iteritems():
        # print(pd.Timestamp(date))



    # print(prices)
    # print(polls)

    return prices['Close'].values, polls['Obama (D)'].values


def load_data_ext():
    return load_data(price_path, polls_path)
# load_data(price_path, polls_path)
