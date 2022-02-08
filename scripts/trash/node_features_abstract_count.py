import argparse
import pandas as pd
import progress_bar as pb

hindex_mean = 10.087608542191562

parser = argparse.ArgumentParser(description="Générer le nombre d'abstract de chaque auteur")
parser.add_argument("authors_abstracts")
parser.add_argument("output")

args = parser.parse_args()

authors_features_file_name = args.authors_features
authors_hindex_name = args.authors_hindex
output_file_name = args.output

df_author_features = pd.read_csv(authors_features_file_name, sep=';', index_col=0)
df_hindex = pd.read_csv(authors_hindex_name, sep=';',index_col=0)

df_output = df_hindex.join(df_author_features, on="author")

df_output.to_csv(output_file_name, sep=";")