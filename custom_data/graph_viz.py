import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random


edgelist= pd.read_csv("coauthorship.csv", sep=';')
g = nx.from_pandas_edgelist(edgelist, 'author_1', 'author_2')

nodes = list(g.nodes)
# random.shuffle(nodes)
gs = g.subgraph(nodes[0:10000])
nx.draw(gs, with_labels=False, node_size=10, alpha=0.5, linewidths=0.2)
plt.show()