import torch
from tqdm import tqdm
from torch.nn import Linear, Dropout

from torch_geometric.nn import GCNConv, GraphSAGE
import torch_geometric.data as data
from torch_cluster import random_walk
from torch_geometric.loader import NeighborLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import networkx as nx

import pandas as pd
import numpy as np
import argparse
import progress_bar as pb

hindex_mean =  10.087608542191562

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"
print('> Using {} device'.format(device))

pb.init(1, _prefix="Initial computation \t \t")

parser = argparse.ArgumentParser(
    description="GCN")
parser.add_argument("node_features",
                    help="Input author_vector correspondance file")
parser.add_argument("graph_metrics")
parser.add_argument("edge_index")
parser.add_argument("train_file", help="Train file")
parser.add_argument("validation_file", help="Validation file")
parser.add_argument("output_file", help="Output submission file")

args = parser.parse_args()

node_features_name = args.node_features
graph_metrics_name = args.graph_metrics
edge_index_name = args.edge_index
train_file_name = args.train_file
validation_file_name = args.validation_file
output_file_name = args.output_file


df_node_features = pd.read_csv(node_features_name, sep=";", index_col=0)
df_graph_metrics = pd.read_csv(graph_metrics_name, sep=";", index_col=0)


df_edge_index = pd.read_csv(edge_index_name, sep=";")

df_train = pd.read_csv(train_file_name, sep=";", index_col=0)
df_validation = pd.read_csv(validation_file_name, sep=";", index_col=0)

df_node_features = pd.concat([df_graph_metrics, df_node_features], ignore_index=True,axis=1)

# to standardise the data

scaler = StandardScaler()
feature_scaled = scaler.fit_transform(df_node_features.values)

df_node_features = pd.DataFrame(feature_scaled, index=df_node_features.index, columns=df_node_features.columns)


# to perform the PCA

# Select the number of principal components we will return
num_components = 100

pca = PCA(n_components=num_components)

principal_components = pca.fit_transform(df_node_features)

df_principal_components = pd.DataFrame(principal_components)

def display_scree_plot(pca):
    '''Display a scree plot for the pca'''

    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="blue",marker='o')
    plt.xlabel("Number of principal components")
    plt.ylabel("Percentage explained variance")
    plt.title("Cumulative explained variance over principal comp.")
    plt.savefig("test.png")

display_scree_plot(pca) 

edge_count = df_edge_index.shape[0]
node_features_count = df_principal_components.shape[1]
node_count = df_principal_components.shape[0]

y = hindex_mean * np.ones(node_count) # True value for train set, mean elswhere
train_mask = np.zeros(node_count) # 1 for train set, 0 elswhere
validation_mask = np.zeros(node_count) # 1 for validation set, 0 elswhere
test_mask = np.ones(node_count) # 1 for test set, 0 elswhere

def compute_train_mask(row):
    y[int(row.name)] = row.hindex
    train_mask[int(row.name)] = 1
    test_mask[int(row.name)] = 0

def compute_validation_mask(row):
    y[int(row.name)] = row.hindex
    validation_mask[int(row.name)] = 1
    test_mask[int(row.name)] = 0

df_train.apply(lambda row: compute_train_mask(row), axis=1)
df_validation.apply(lambda row: compute_validation_mask(row), axis=1)

ts_node_features = torch.tensor(df_principal_components.values).float()  # Tensor with node features

ts_edge_index = torch.tensor(df_edge_index.values).long()  # tensor with node links

ts_y = torch.tensor(y).float()

ts_train_mask = torch.tensor(train_mask).bool()
ts_validation_mask = torch.tensor(validation_mask).bool()
ts_test_mask = torch.tensor(test_mask).long()

ts_node_features = ts_node_features.view(node_count, node_features_count)
ts_edge_index = torch.transpose(ts_edge_index, 1, 0)

graph_data = data.Data(x=ts_node_features, edge_index=ts_edge_index, y=ts_y, train_mask=ts_train_mask, test_mask=ts_test_mask, validation_mask=ts_validation_mask)


print()
print("> Summary : ")
print(">> Num node features", graph_data.num_node_features)
print(">> Num node", graph_data.num_nodes)
print(">> Num edge features", graph_data.num_edge_features)
print(">> Num edge", graph_data.num_edges)
print(">> Num features", graph_data.num_features)
print(">> Directed : ", graph_data.is_directed())

inner_layer_size_1 = 65
inner_layer_size_2 = 64
inner_layer_size_3 = 1

class GCN(torch.nn.Module):

    def __init__(self):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(node_features_count, inner_layer_size_1)
        self.conv2 = GCNConv(inner_layer_size_1, inner_layer_size_2)
        #self.conv3 = SAGEConv(inner_layer_size_2, inner_layer_size_3)
        self.out = Linear(inner_layer_size_2, 1)

        self.dropout = Dropout(p=0.1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = torch.relu(x)

        """x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = torch.tanh(x)"""

        x = self.out(x)
        x = torch.relu(x)

        return x

class GSAGE(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(42)
        super().__init__()
        self.conv = GraphSAGE(node_features_count,64,2,64,dropout=0.1,jk='lstm')
        self.linear1 = torch.nn.Linear(64,64)
        self.linear2 = torch.nn.Linear(64,1)

    def forward(self, data):
      x, edge_index = data.x, data.edge_index
      # GraphSAGE network
      x = self.conv(x, edge_index)

      # Multi-layer perceptron
      x = self.linear1(x)
      x = torch.nn.functional.relu(x)
      x = torch.nn.functional.dropout(x, p=0.2)
      x = self.linear2(x)
      
      return torch.nn.functional.relu(x)

model = GCN()

print()
print("> Model : ", model)

# Use GPU for training

learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=decay)
loss_fn = torch.nn.MSELoss(reduction='mean')
losses = []

model = model.to(device)
graph_data = graph_data.to(device)

print()
print("> Starting training")

def train():
    """model.train()

    '''total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)[:batch_size]
        loss = loss_fn(out.flatten(), batch.y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples"""
    model.train()
    optimizer.zero_grad()
    # Use all data as input, because all nodes have node features

    out = model(graph_data.x, graph_data.edge_index)
    out = torch.transpose(out, 0, 1).flatten()

    # Only use nodes with labels available for loss calculation --> mask
    loss = loss_fn(out[graph_data.train_mask],
                   graph_data.y[graph_data.train_mask])


    loss.backward()
    optimizer.step()
    return loss.item()

def validate():
    model.eval()
    out = model(graph_data.x, graph_data.edge_index)
    out = torch.transpose(out, 0, 1).flatten()
    loss = loss_fn(out[graph_data.validation_mask], graph_data.y[graph_data.validation_mask])

    return loss.item()

def test_raw():
    model.eval()
    out = model(graph_data.x, graph_data.edge_index)

    return out

def test():
    out = test_raw()
    out = torch.transpose(out, 0, 1).flatten()

    return out

# for epoch in tqdm(range(301)):
for epoch in range(1001):
    loss = train()
    losses.append(loss)
    if (epoch % 100 == 0):
        validation_loss = validate()
        print(f'Epoch: {epoch} \t \t Loss: {loss} \t \t Validation loss : {validation_loss}')    
    
result = test()

df_output = pd.DataFrame()
df_output.insert(0, "author", df_node_features.index)
df_output.insert(1, "hindex", result.cpu().detach().numpy())

df_output.to_csv(output_file_name, sep=",", index=False)

