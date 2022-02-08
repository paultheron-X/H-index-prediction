import networkx as nx
import pandas as pd
import argparse
import progress_bar as pb

pb.init(1, _prefix="Refactoring metrics \t \t")

parser = argparse.ArgumentParser(
    description="GCN")
parser.add_argument("edge_index")
parser.add_argument("author_abstract_count")
parser.add_argument("output")

args = parser.parse_args()

edge_index_file_name = args.edge_index
author_abstract_count_name = args.author_abstract_count
output_file_name = args.output

df_edge_index = pd.read_csv(edge_index_file_name, sep=";")
df_author_abstract_count = pd.read_csv(author_abstract_count_name, sep=";", index_col=0)
df_graph_metrics = pd.DataFrame()

### Calcul des features du graph

G = nx.from_pandas_edgelist(df_edge_index, source= 'author_2', target= 'author_1')

degree = nx.degree(G)
core_number = nx.core_number(G)
degree_centr = nx.degree_centrality(G)
nbr_avg_deg = nx.average_neighbor_degree(G)
pagerank = nx.pagerank(G, alpha=0.9)
olayer = nx.onion_layers(G)

df_graph_metrics.insert(0, "degree", degree)
df_graph_metrics.insert(1, "core_number", core_number)
df_graph_metrics.insert(2, "degree_centr", degree_centr)
df_graph_metrics.insert(3, "nbr_avg_deg", nbr_avg_deg)
df_graph_metrics.insert(4, "pagerank", pagerank)
df_graph_metrics.insert(5, "olayer", olayer)

pb.set_length(df_graph_metrics.shape[0])

def compute_row_feature(row):
    pb.progress(row.name)
    row.degree = G.degree[row.name]
    row.core_number = core_number[row.name]
    row.degree_centr = degree_centr[row.name]
    row.nbr_avg_deg = nbr_avg_deg[row.name]
    row.pagerank = pagerank[row.name]
    row.olayer = olayer[row.name]
    return row

df_graph_metrics = df_graph_metrics.apply(lambda x: compute_row_feature(x), axis=1)
df_graph_metrics.index.name="author"

df_graph_metrics.insert(0, "author_abstract_count", df_author_abstract_count["count"])

pb.progress(df_graph_metrics.shape[0])
print(df_graph_metrics)

df_graph_metrics.to_csv(output_file_name, sep=";",index=True)