import random
import math
import numpy as np

from pomegranate import *
import load_data
from matplotlib import pyplot as plt
from naive_bayes import get_naive_bayes

# data paths
price_path = 'data/2012_election_prices.csv'
polls_path = 'data/2012_election_polls.csv'

# We assume that we are attempting to predict a binary outcome.
# Of couse we can always take a n-ary outcome to be a series of
# binary choices.

# we take timesteps to be points where we have polls. This makes the
# assumption that time since prev poll is not relevant, which is
# reasonable because we can assume not much interesting happened
# in the election if there are not polls being taken. 

# number of agents in the market.
n = 5

# get_data() should return a list of the probabilities of the
# payoff outcome given by the climate data.
data = load_data.load(price_path, polls_path)[:665]
print len(data)
climate_diffs = data["Obama (D)"] / (data["Obama (D)"] + data["Romney (R)"])

T = len(climate_diffs)

max_noise_eps = 0.28 # noise introduced into the climate signal for each agent

e = 0.58  # weight we give to new polls vs previous price

def majority(n_a):
    table = []
    table_minus = []
    for i in range(1<<n_a):
        table += [["P" if (i>>j) % 2 == 0 else "N" for j in range(n_a)]]
    for row in table:
        c = row.count("P")
        if c > n_a/2.0:
            table_minus += [row + ["N", 0]]
            row += ["P", 1]
        elif c < n_a/2.0:
            table_minus += [row + ["P", 0]]
            row += ["N", 1]
        else:
            table_minus += [row + ["P", 0.5]]
            row += ["P", 0.5]
    return table + table_minus

def get_price_preds(e=e, max_noise_eps=max_noise_eps, stupid=0):
    print e
    model = BayesianNetwork("Prediction Market")
    prices = []
    # "P" is getting the payoff, "N" is not, but they are interchangeable in this model.
    climate_dists = []
    for p in climate_diffs:
        dist = DiscreteDistribution({"P": p, "N": 1-p})
        node = Node(dist, name="climate_prob")
        model.add_state(node)
        climate_dists += [(dist, node)]
    
    # print climate_dists
    
    climate_signals = []
    
    agents = []
    for i in range(T):
        timestep_signals = []
        timestep_agent = []
        for a in range(n): 
            P_noise = random.random() * max_noise_eps
            N_noise = random.random() * max_noise_eps
            dist = ConditionalProbabilityTable([
                    ["P", "P", 1 - P_noise],
                    ["P", "N", P_noise],
                    ["N", "P", N_noise],
                    ["N", "N", 1 - N_noise]], [climate_dists[i][0]])
            node = Node(dist, name="climate_prob")
            model.add_state(node)
            model.add_edge(climate_dists[i][1], node)
            timestep_signals += [(dist, node)]
            if i == 0:
                dist = ConditionalProbabilityTable(   [
                ["P", "P", 1],
                ["P", "N", 0],
                ["N", "N", 1],
                ["N", "P", 0],
                ["P", "P", 1],
                ["P", "N", 0],
                ["N", "N", 1],
                ["N", "P", 0]
            ], [timestep_signals[a][0]])
            else:
                if a < stupid:
                    dist = ConditionalProbabilityTable(   [
                        ["P", "P", "N", e],
                        ["P", "P", "P", 1 - e],
                        ["P", "N", "N", e],
                        ["P", "N", "P", 1 - e],
                        ["N", "P", "N", 1 - e],
                        ["N", "P", "P", e],
                        ["N", "N", "N", 1 - e],
                        ["N", "N", "P", e]
                    ], [prices[i-1][0], timestep_signals[a][0]])
                else:
                    dist = ConditionalProbabilityTable(   [
                        ["P", "P", "N", 0],
                        ["P", "P", "P", 1],
                        ["P", "N", "N", e],
                        ["P", "N", "P", 1 - e],
                        ["N", "P", "N", 1 - e],
                        ["N", "P", "P", e],
                        ["N", "N", "N", 1],
                        ["N", "N", "P", 0]
                    ], [prices[i-1][0], timestep_signals[a][0]])
            node = Node(dist, name="agent"+str(i)+str(a))
            model.add_state(node)
            if i != 0:
                model.add_edge(prices[i-1][1], node)
            model.add_edge(timestep_signals[a][1], node)
            timestep_agent += [(dist, node)]
        agents += [timestep_agent]
        climate_signals += [timestep_signals]
    
        price_dist = ConditionalProbabilityTable(majority(len(timestep_agent)), [d_ for d_, n_ in timestep_agent])
        price_node = Node(price_dist, name="price"+str(i))
        model.add_state(price_node)
        for d_, n_ in timestep_agent:
            model.add_edge(n_, price_node)
        prices += [(price_dist, price_node)]
    
    # print model
    model.bake()
    # print model
    # print len(model.states)
    prices_indexes = []
    for i, state in enumerate(model.states):
        if "price" in state.name:
            prices_indexes += [i]
    
    # nx.draw(model.graph)
    
    # print model.probability("time1")
    pred = model.predict_proba({}, max_iterations=100)
    prices = [pred[ix].values()[0] for ix in prices_indexes]
    return prices

# takes total KL divergence; each is a vector of Bernoulli rv's of the first item
# dist1 is P, dist2 is Q
def KL(dist1, dist2):
    total = 0
    for p1, p2 in zip(dist1, dist2):
        total += (p1*math.log(p1/p2) + (1-p1)*math.log((1-p1)/(1-p2)))
    return total

if __name__ == "__main__":
    # print [p.values()[0] for p in prices]
  #  prices = get_price_preds()
  #  plt.plot(prices)
  #  # print len(prices)
  #  # print len(data["Close"])
  #  plt.plot(data["Close"]/100)
  #  print "divergence: ", KL(prices, data["Close"]/100)
  #  plt.plot(get_naive_bayes())
  #  # plt.show()
  #  plt.clf()
    ms = [0, 0.03 0.06, 0.08, 0.1,  0.16, 0.22, 0.28, 0.34, 0.4,  0.46, 0.52, 0.58, 0.64]
    kls = []
    for m in ms:
        prices = get_price_preds(e=0.58, stupid=0, max_noise_eps=m)
        # plt.plot(prices)
        # plt.plot(data["Close"]/100)
        # plt.savefig(str(s)+"_#2.png")
        # plt.clf()
        print m
        kls += [KL(prices, data["Close"]/100)]
        print kls
    print kls
    print ms
    plt.plot(kls, ms)
    plt.savefig("m.png")
    # print model

