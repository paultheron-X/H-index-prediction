import numpy as np
from gensim.models import KeyedVectors
import argparse


parser = argparse.ArgumentParser(
    description="Générer un vecteur à partir d'un modèle sauvegardé")
parser.add_argument("model_file", help="W2V model file")
parser.add_argument(
    "-v", help="Model (and so output) vector size", default=100, type=int)

args = parser.parse_args()

model_file_name = args.model_file
vector_size = args.v

# Init progress bar

wv = KeyedVectors.load_word2vec_format(model_file_name, binary=True)

print("> Model loaded")
while(True):
    action = input("Find nearest : n, Get vec : v, Find nearest from vec : nv, Leave : q => ")
    if (action == 'q'):
        exit()
    elif (action == 'n'):
        test_word = input("Input word : ")
        try:
            print(wv.most_similar(positive=[test_word], topn=10))
        except KeyError:
            print("Word doesn't exist in dataset")
            continue
    elif(action == 'v'):
        test_word = input("Input word : ")
        try:
            print(wv[test_word])
        except KeyError:
            print("Word doesn't exist in dataset")
            continue
    elif(action == 'nv'):
        test_vec = input("Input vector (sep = ;) : ")
        try:
            vec = np.fromstring(test_vec, count=vector_size, sep=';')
            print(wv.most_similar(positive=[vec], topn=10))
        except KeyError:
            print("Word doesn't exist in dataset")
            continue
        except ValueError:
            print("Not proper array")
            continue
