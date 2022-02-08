import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import progress_bar as pb
import argparse


parser = argparse.ArgumentParser(
    description="Générer un vecteur à partir d'un modèle sauvegardé")
parser.add_argument("input_file", help="Input .csv file of abstracts")
parser.add_argument(
    "output_file", help="Output .csv file of vectors for each abstract's ID")
parser.add_argument("model_file", help="W2V model file")
parser.add_argument(
    "-v", help="Model (and so output) vector size", default=100, type=int)

args = parser.parse_args()

input_file_name = args.input_file
output_file_name = args.output_file
input_model_name = args.model_file
vector_size = args.v

# Init progress bar
pb.init(1, _prefix='Computing vectors \t \t', _suffix='Complete')
pb.progress(0)

wv = KeyedVectors.load_word2vec_format(input_model_name, binary=True)

df_input = pd.read_csv(input_file_name, sep=";", encoding='utf8')
rows_count = df_input.shape[0]

df_vector = pd.DataFrame(columns=[str(i) for i in range(vector_size)])
df_input = pd.concat([df_input, df_vector], ignore_index=True)

pb.set_length(rows_count - 1 + 1000)


def compute_vector(row):
    words = row["abstract"].split(" ")
    count = len(words)
    sum = np.zeros(vector_size)
    for word in words:
        try:
            vector = wv[word]
            sum += vector
        except KeyError:
            continue
            # print(word)
    mean = sum / count
    pb.progress(row.name)
    return list(mean)


df_output = df_input.apply(lambda row: compute_vector(
    row), axis=1, result_type='expand')

df_output.insert(0, "ID", df_input["ID"])

pb.progress(rows_count - 1 + 500)

df_output.to_csv(output_file_name, sep=";", index=False)

pb.progress(rows_count - 1 + 1000)
