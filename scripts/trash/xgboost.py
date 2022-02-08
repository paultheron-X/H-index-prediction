import os
from networkx.algorithms.centrality.degree_alg import degree_centrality
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import GradientBoostingClassifier

# read training data
df_train = pd.read_csv('../raw_data/train.csv', dtype={'author': np.int64, 'hindex': np.float32})
n_train = df_train.shape[0]

# read test data
df_test = pd.read_csv('../raw_data/test.csv', dtype={'author': np.int64})
n_test = df_test.shape[0]

# load the graph    
G = nx.read_edgelist('../raw_data/coauthorship.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges() 
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)


# computes structural features for each node
core_number = nx.core_number(G)
degree_centr = nx.degree_centrality(G)
nbr_avg_deg = nx.average_neighbor_degree(G)
pagerank = nx.pagerank(G, alpha=0.9)
olayer = nx.onion_layers(G)

# create the training matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number 
X_train = np.zeros((n_train, 6))
y_train = np.zeros(n_train)
for i,row in df_train.iterrows():
    node = row['author']
    X_train[i,0] = G.degree(node)
    X_train[i,1] = core_number[node]
    X_train[i,2] = degree_centr[node]
    X_train[i,3] = nbr_avg_deg[node]
    X_train[i,4] = pagerank[node]
    X_train[i,5] = olayer[node]

    y_train[i] = row['hindex']

print(X_train)
# create the test matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number
X_test = np.zeros((n_test, 6))
for i,row in df_test.iterrows():
    node = row['author']
    X_test[i,0] = G.degree(node)
    X_test[i,1] = core_number[node]
    X_test[i,2] = degree_centr[node]
    X_test[i,3] = nbr_avg_deg[node]
    X_test[i,4] = pagerank[node]
    X_test[i,5] = olayer[node]
    
# train a regression model and make predictions
reg = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
print("Starting training")
reg.fit(X_train, y_train)
print("Starting predicting")
y_pred = reg.predict(X_test)

# write the predictions to file
df_test['hindex'] = pd.Series(np.round_(y_pred, decimals=3))

df_test.loc[:,["author","hindex"]].to_csv('submission.csv', index=False)





