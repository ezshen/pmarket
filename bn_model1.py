from pomegranate import *
import networkx as nx

# We assume that we are attempting to predict a binary outcome.
# Of couse we can always take a n-ary outcome to be a series of
# binary choices.

# number of agents in the market.
n = 5

# get_data() should return a list of the probabilities of the
# payoff outcome given by the climate data.
# climate_probs = get_data()
climate_probs = [0.4, 0.5, 0.6, 0.7]
T = len(climate_probs)

noise_eps = 0.1 # noise introduced into the climate signal for each agent
e = 0.2  # weight we give to new polls vs previous price
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
for p in climate_probs:
    dist = DiscreteDistribution({"P": p, "N": 1-p})
    node = Node(dist, name="climate_prob")
    model.add_state(node)
    climate_dists += [(dist, node)]

climate_signals = []
for i in range(T):      # TODO needs to be for each agent
    dist = ConditionalProbabilityTable([
            ["P", "P", 1 - noise_eps],
            ["P", "N", noise_eps],
            ["N", "P", noise_eps],
            ["N", "N", 1 - noise_eps]], [climate_dists[i][0]])
    node = Node(dist, name="climate_prob")
    model.add_state(node)
    model.add_edge(climate_dists[i][1], node)
    climate_signals += [(dist, node)]

agents = []
for i in range(T):
    timestep_arr = []
    for a in range(n):
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
        ], [climate_signals[i][0]])
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
            ], [prices[i-1][0], climate_signals[i][0]])
        node = Node(dist, name="agent")
        model.add_state(node)
        if i != 0:
            model.add_edge(prices[i-1][1], node)
        model.add_edge(climate_signals[i][1], node)
        timestep_arr += [(dist, node)]
    agents += [timestep_arr]

    price_dist = ConditionalProbabilityTable(majority(len(timestep_arr)), [d_ for d_, n_ in timestep_arr])
    price_node = Node(price_dist, name="time1")
    model.add_state(price_node)
    for d_, n_ in timestep_arr:
        model.add_edge(n_, price_node)
    prices += [(price_dist, price_node)]

print model
model.bake()

# nx.draw(model.graph)

# print model.probability("time1")
print model.predict_proba({})
# print model














