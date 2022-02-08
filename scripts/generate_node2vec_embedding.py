import networkx as nx
import pandas as pd
import os
import argparse
from node2vec import Node2Vec
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(
    description="Générer un vecteur à partir d'un modèle sauvegardé")
parser.add_argument("input_file", help="Input .edgelist file")
parser.add_argument(
    "output_file", help="Output .csv file of vectors for each node's embedding")
parser.add_argument("model_file", help="Output .model file of node2vec embedding")
parser.add_argument(
    "-v", help="Model (and so output) vector size", default=100, type=int)

args = parser.parse_args()


input_file_name = args.input_file
output_file_name = args.output_file
model_output_file = args.model_file
vector_size_ = args.v

df_edge_index = pd.read_csv(input_file_name, sep=";")
G = nx.from_pandas_edgelist(df_edge_index, source= 'author_2', target= 'author_1')
#G = nx.fast_gnp_random_graph(n=100, p=0.5)
#g2v = Node2Vec(n_components=vector_size_, walklen=60, threads=os.cpu_count(), w2vparams={'window': 10, 'negative':5, 'iter': 20, 'batch_words':128})
node2vec = Node2Vec(G, dimensions=100, walk_length=30, num_walks=15, workers=os.cpu_count())  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)

emb_per_nodes = {}
for node in tqdm(list(G.nodes())):
    emb_per_nodes[node] = model.wv[node]

df_output = pd.DataFrame.from_dict(emb_per_nodes, orient="index")
df_output.to_csv(output_file_name, sep=";", index=True)

model.save(model_output_file)