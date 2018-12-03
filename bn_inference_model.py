import random

from pomegranate import *
import load_data
from matplotlib import pyplot as plt

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
# number of time steps in the market
T = 20
# known prices (so far)
initial_diffs = [0.5]   # prior on the first poll result
prices_ = [0.5, 0.48, 0.55, 0.48, 0.53, 0.505, 0.501, 0.48]
# random walk parameter for climate
w = 0.2

# data = load_data.load(price_path, polls_path)
# print len(data)
# climate_diffs = data["Obama (D)"] / (data["Obama (D)"] + data["Romney (R)"])

# T = len(climate_diffs)

max_noise_eps = 0.08 # noise introduced into the climate signal for each agent

e = 0.58  # weight we give to new polls vs previous price
prices = []

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

model = BayesianNetwork("Prediction Market")

# "P" is getting the payoff, "N" is not, but they are interchangeable in this model.
climate_dists = []
for i, p in enumerate(initial_diffs):
    # TODO scale by normal distribution??? I feel like this will probably break it...
    dist = DiscreteDistribution({"P": p, "N": 1-p})
    node = Node(dist, name="climate_prob_spec"+str(i))
    model.add_state(node)
    climate_dists += [(dist, node)]

for i in range(len(initial_diffs), T):
    dist = ConditionalProbabilityTable([
            ["P", "P", 1 - w],
            ["P", "N", w],
            ["N", "P", w],
            ["N", "N", 1 - w]], [climate_dists[i-1][0]])
    node = Node(dist, name="climate_prob_walk"+str(i))
    model.add_state(node)
    model.add_edge(climate_dists[i-1][1], node)
    climate_dists += [(dist, node)]

# print climate_dists

climate_signals = []
agents = []
for i in range(T):
    timestep_signals = []
    timestep_agent = []
    for a in range(n):    # TODO needs to have bias, not just variance
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
last_climate_index = -1
for i, state in enumerate(model.states):
    if "price" in state.name:
        prices_indexes += [i]
    if "climate_prob_spec"+str(T-1) in state.name:
        last_climate_index = i

# nx.draw(model.graph)
# print model.probability("time1")
pred = model.predict_proba({"price" + str(i): DiscreteDistribution({"P": pr, "N": 1-pr}) for i, pr in enumerate(prices_)})
prices = [pred[ix].values()[0] for ix in prices_indexes]
print "final climate: ", pred[last_climate_index]
# print [p.values()[0] for p in prices]
plt.plot(prices)
# print len(prices)
# print len(data["Close"])
plt.plot(prices_)
plt.show()
# print model


