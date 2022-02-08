from logging import log
import pandas as pd
import numpy as np
import progress_bar as pb
import argparse


parser = argparse.ArgumentParser(
    description="Générer les features de chaque auteur (moyenne de ses publications)")
parser.add_argument("author_abstract_file",
                    help="Input .csv file of author <-> abstracts correspondance")
parser.add_argument("abstract_vectors_file",
                    help="Input .csv file of abstracts")
parser.add_argument(
    "output_file", help="Output .csv file of author's attributes")
parser.add_argument(
    "-v", help="Model (and so output) vector size", default=20, type=int)

args = parser.parse_args()

abstract_vectors_file_name = args.abstract_vectors_file
author_abstract_file_name = args.author_abstract_file
output_file_name = args.output_file
vector_size = args.v

# Init progress bar
pb.init(1, _prefix='Computing vectors \t \t', _suffix='Complete')
pb.progress(0)


df_abstract_vector = pd.read_csv(
    abstract_vectors_file_name, sep=";", encoding='utf8')
df_abstract_vector.set_index("ID", inplace=True)

df_author_abstract = pd.read_csv(
    author_abstract_file_name, sep=";", encoding='utf8')


author_count = df_author_abstract.shape[0]
pb.set_length(author_count)

counter = 0
coefs = (2, 1.8, 1.6, 1.4, 1.2)
def compute_row(row):
    global counter
    pb.progress(counter)
    counter += 1

    mean = np.zeros(vector_size)
    count = 0
    for i in range(1, 6):
        try:

            mean += coefs[i - 1] * df_abstract_vector.loc[int(row["paper_" + str(i)])].to_numpy()
            count += coefs[i - 1]
        except (ValueError, KeyError):
            continue
    if (count == 0):
        return mean
    else:
        return mean / count


df_output = df_author_abstract.apply(
    lambda x: compute_row(x), result_type='expand', axis=1)

df_output.insert(0, "author", df_author_abstract["author"])

df_output.to_csv(output_file_name, sep=";", index=False)

pb.progress(author_count)
