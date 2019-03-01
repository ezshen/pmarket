import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from naive_bayes import get_naive_bayes
from bn_model1 import KL
import load_data

price_path = 'data/2012_election_prices.csv'
polls_path = 'data/2012_election_polls.csv'

data = load_data.load(price_path, polls_path)[:665]
print len(data)
climate_diffs = data["Obama (D)"] / (data["Obama (D)"] + data["Romney (R)"])

probs = []
for d in climate_diffs:
	probs += [stats.norm.cdf(0.5, loc=0.5 - (d-0.5), scale=1/np.sqrt(800))]

price_preds = np.load("prices.npy")
one, = plt.plot(price_preds, label='DBN predictions')
two, = plt.plot(data["Close"]/100, label='Actual prices')
nb = get_naive_bayes()
three, = plt.plot(nb, label='Bayesian Updating')
n = 30
kernel = np.array([1./n for _ in range(n)])
smoothed_prices = np.convolve(price_preds, kernel, mode='valid')
four, = plt.plot(np.arange((n+1)/2, len(price_preds) - (n+2)/2), smoothed_prices[:-2], ls="dashed", label='Smoothed DBN')

plt.legend(handles=[one, two, three, four])
plt.savefig("all_baselines.png")

print "KLs:"
print "BU", KL(nb, data["Close"]/100)
print "ours", KL(price_preds, data["Close"]/100)
print "polls", KL(probs, data["Close"]/100)

