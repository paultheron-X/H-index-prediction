import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt


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

core_number = nx.core_number(G)

    
def calcule(x):
    #return core_number[x['author']]
    return G.degree(x['author'])


    #return np.log(G.degree(x['author']) +1)

#df_output = df_train.apply( lambda x: calcule(x), result_type = 'broadcast', axis=1)

df_train.insert(2, "neigbors", df_train.apply( lambda x: calcule(x), result_type = 'expand', axis=1))

print(df_train)

# to show the 
sns.set(font_scale = 1.3)
sns.set_style("whitegrid", {'axes.grid' : False})
jp = sns.jointplot(data=df_train, x="neigbors", y="hindex", 
              kind="kde", fill = True, xlim = [0,65], ylim = [0,65],
               space = 0, marginal_ticks = False, palette = "rocket"
             )
jp.set_axis_labels("Node degree", "H-index")

plt.savefig('cor.png')

'''
# computes structural features for each node
core_number = nx.core_number(G)

# create the training matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number 
X_train = np.zeros((n_train, 2))
y_train = np.zeros(n_train)
for i,row in df_train.iterrows():
    node = row['author']
    X_train[i,0] = G.degree(node)
    X_train[i,1] = core_number[node]
    y_train[i] = row['hindex']

# create the test matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number
X_test = np.zeros((n_test, 2))
for i,row in df_test.iterrows():
    node = row['author']
    X_test[i,0] = G.degree(node)
    X_test[i,1] = core_number[node]
    
# train a regression model and make predictions
reg = Lasso(alpha=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# write the predictions to file
df_test['hindex'] = pd.Series(np.round_(y_pred, decimals=3))

df_test.loc[:,["author","hindex"]].to_csv('submission.csv', index=False)'''





