from pomegranate import *

# We assume that we are attempting to predict a binary outcome.
# Of couse we can always take a n-ary outcome to be a series of
# binary choices.

# number of agents in the market.
n = 5

# get_data() should return a list of the probabilities of the
# payoff outcome given by the climate data.
# climate_probs = get_data()
climate_probs = [0.4, 0.5, 0.4, 0.45, 0.6, 0.65, 0.75, 0.7]
T = len(climate_probs)

noise_eps = 0.1
prices = []

def agents(dists):
    table = []
    n_a = len(dists)
    for i in range(1<<n_a):
        table += [[1<<i]]

# "P" is getting the payoff, "N" is not, but they are interchangeable in this model.
climate_nodes = [DiscreteDistribution({"P": p, "N": 1-p}) for p in climate_probs]
climate_signals = [
        ConditionalProbabilityTable([
            ["P", "P", 1 - noise_eps],
            ["P", "N", noise_eps],
            ["N", "P", noise_eps],
            ["N", "N", 1 - noise_eps]
        ], [climate_nodes[i]])
    for i in range(T)
]

agents = []
for i in range(T):
    agents.append([
    [   [
            {("P", "P", "N"): 0,
            ("P", "P", "P"): 1,
            ("P", "N", "N"): 1/(a+1), # this is almost surely the wrong way to do this
            ("P", "N", "P"): 1 - 1/(a+1),
            ("N", "P", "N"): 1 - 1/(a+1),
            ("N", "P", "P"): 1/(a+1),
            ("N", "N", "N"): 1,
            ("N", "N", "P"): 0},
        ], prices[i-1] if i != 0 else UniformDistribution(n), climate_signals[a] for a in range(n)]
    ])
    prices.append(majority(agents[-1]))














