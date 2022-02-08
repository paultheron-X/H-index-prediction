import pandas as pd
import progress_bar as pb
import argparse


parser = argparse.ArgumentParser(
    description="Générer un vecteur à partir d'un modèle sauvegardé")
parser.add_argument("authors_vectors_file", help="Input author_vector correspondance file")
parser.add_argument("output_dictionnary_file", help="Output dictionnary file")

args = parser.parse_args()

authors_vectors_file_name = args.authors_vectors_file
output_dictionnary_file_name = args.output_dictionnary_file

# Init progress bar
pb.init(1, _prefix='Generating dictionnary \t \t', _suffix='Complete')
pb.progress(0)

df_author_vector = pd.read_csv(authors_vectors_file_name, sep=";")
df_dictionnary = df_author_vector[["author"]]
df_dictionnary.reset_index(inplace=True)
df_dictionnary = df_dictionnary.rename(columns = {'index':'new_author'})
df_dictionnary = df_dictionnary.set_index("author")

df_dictionnary.to_csv(output_dictionnary_file_name, sep=";", index=True)

pb.progress(1)
