import pandas as pd

price_path = 'data/2012_election_prices.csv'
polls_path = 'data/2012_election_polls.csv'

def load_data(price_path, polls_path):
    prices = pd.read_csv(price_path, sep='\t', dtype={u'Date': str, u'Open': float, u'High': float, u'Low Close': float, u'Volume': int}, index_col=False)
    polls = pd.read_csv(polls_path, sep='\t', dtype={u'Poll': str, u'Date': str, u'Sample': str, u'MoE': str, u'Obama (D)': float, u'Romney (R)': float, u'Spread': str}, index_col=False)
    return prices, polls

load_data(price_path, polls_path)
