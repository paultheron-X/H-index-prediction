import gensim
from gensim.models import Word2Vec
from gensim.test.utils import datapath
import pandas as pd
import argparse
import progress_bar as pb
import os

tmp_file_name = "tmp"

parser = argparse.ArgumentParser(description="Entraine et sauvegarde un w2v")
parser.add_argument("input_file", help="Input .csv file of built abstracts")
parser.add_argument(
    "output_file", help="Output .wordvectors file for the model")
parser.add_argument("-v", default=100,
                    help="Size of the output vector", type=int)

args = parser.parse_args()

input_file_name = args.input_file
output_model_name = args.output_file
vector_size = args.v

pb.init(10, _prefix="Entrainement \t \t \t")
pb.progress(0)


df_input = pd.read_csv(input_file_name, sep=";")

tmp_file = open(tmp_file_name, 'w', encoding='UTF8')

df_input.apply(lambda row: tmp_file.write(row["abstract"] + "\n"), axis=1)
pb.progress(1)

tmp_file.close()

if (gensim.__version__ >='4.0.0'): # Not all versions of gensim have the same arguments. We guess that a change append between 3.x to 4.x
    model = Word2Vec(corpus_file=tmp_file_name, sg=1, vector_size=vector_size, window=5, workers=4)
else:
    model = Word2Vec(corpus_file=tmp_file_name, sg=1, size=vector_size, window=5, workers=4)

pb.progress(9)
model.wv.save_word2vec_format(output_model_name, binary=True)
os.remove(tmp_file_name)
pb.progress(10)
