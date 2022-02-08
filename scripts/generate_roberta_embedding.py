import argparse
from sentence_transformers import SentenceTransformer
from string import digits, ascii_letters, punctuation, printable
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Générer un vecteur à partir d'un modèle sauvegardé")
parser.add_argument("input_file", help="Input .csv file of abstracts")
parser.add_argument(
    "output_file", help="Output .csv file of vectors for each abstract's ID")
parser.add_argument(
    "-v", help="Model (and so output) vector size", default=100, type=int)

args = parser.parse_args()


input_file_name = args.input_file
output_file_name = args.output_file
vector_size_ = args.v

df_abstract = pd.read_csv(input_file_name, sep = ";", index_col=0)


valid = ascii_letters + digits + punctuation + printable
paper_id = []
text = []

for index,row in tqdm(df_abstract.iterrows(), total=df_abstract.shape[0]):
    txt = ''.join([char for char in row.abstract if char in valid])
    if len(txt) > 0:
        paper_id.append(index)
        text.append(txt)

model = SentenceTransformer('stsb-roberta-base')
model.cuda()
embeddings = model.encode(text)

emb_per_paper = {}
for idx, id in enumerate(paper_id):
    emb_per_paper[id] = embeddings[idx]

df_output = pd.DataFrame.from_dict(emb_per_paper, orient="index")
df_output.to_csv(output_file_name, sep=";", index=True)


