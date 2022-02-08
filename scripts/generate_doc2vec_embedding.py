import argparse
import nltk
from nltk.corpus import stopwords 
from tqdm import tqdm
from string import digits, ascii_letters, punctuation, printable
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


nltk.download('stopwords')

parser = argparse.ArgumentParser(
    description="Générer doc2vec")
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
    else:
        paper_id.append(index)
        text.append("science")


stop_words = set(stopwords.words('english')) 
doc = []
for txt in tqdm(text):
    p = txt.split()
    p_clean = [l for l in p if l not in stop_words]
    doc.append(p_clean)
del text

tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(doc)]
print('> Preprocessing finished')
model = Doc2Vec(tagged_data, vector_size = vector_size_, window = 5, min_count = 2, epochs = 100, workers=10)

print('> Processing finished')

emb_per_paper = {}
for idx, id_ in tqdm(enumerate(paper_id)):
    emb_per_paper[id_] = model.docvecs[idx]

df_output = pd.DataFrame.from_dict(emb_per_paper, orient="index")
df_output.to_csv(output_file_name, sep=";", index=True)
