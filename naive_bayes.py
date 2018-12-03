import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from pomegranate import *
import load_data

def get_naive_bayes():
    price_path = 'data/2012_election_prices.csv'
    polls_path = 'data/2012_election_polls.csv'
    
    data = load_data.load(price_path, polls_path)
    print data
    climate_diffs = data["Obama (D)"] - data["Romney (R)"]
    T = climate_diffs.shape[0]
    
    N_st = 1000
    prior_delta = 0
    h = 0.25    # TODO what should the prior precision be
    
    k_ts = np.power(N_st, 0.5)
    
    m = np.zeros(T)
    v = np.zeros(T)
    for t in range(T):
    	m[t] = h/(t * k_ts + h)*prior_delta + np.sum([climate_diffs[i]*k_ts/(t * k_ts + h) for i in range(t)])
    	v[t] = t * k_ts + h
    
    p = np.zeros(T)
    for t in range(T):
    	p[t] = 1 - stats.norm.cdf(-m[t]/np.power(v[t], 0.5))
    
    return p

if __name__ == "__main__":
	p = get_naive_bayes()
	plt.plot(p)
	plt.show()
