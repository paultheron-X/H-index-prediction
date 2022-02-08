import pandas as pd
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser(
    description="Générer un vecteur à partir d'un modèle sauvegardé")
    
parser.add_argument("input_normal")
parser.add_argument("input_doc2vec")
parser.add_argument("train_file")
parser.add_argument("validation_file")
parser.add_argument("test_file")
parser.add_argument("output_file")

args = parser.parse_args()

input_normal_name = args.input_normal
input_doc2vec_name = args.input_doc2vec
train_file_name = args.train_file
validation_file_name = args.validation_file
test_file_name = args.test_file
output_file_name = args.output_file

df_embeddings = pd.read_csv(input_normal_name, sep=",", index_col=0)
df_hindex = pd.read_csv(input_doc2vec_name, sep=",", index_col=0)

df_train = pd.read_csv(train_file_name, sep=";", index_col=0)
df_validation = pd.read_csv(validation_file_name, sep=";", index_col=0)
df_test = pd.read_csv(test_file_name, sep=";", index_col=0)

df_input = df_embeddings
df_input.insert(1, "from_hindex", df_hindex["hindex"])
       
X_train = df_input.loc[df_train.index].to_numpy()
X_validation = df_input.loc[df_validation.index].to_numpy()
X_test = df_input.loc[df_test.index].to_numpy()
X_all = np.concatenate([X_train, X_validation, X_test])
y_train = df_train["hindex"].to_numpy()
y_validation = df_validation["hindex"].to_numpy()

reg = LinearRegression().fit(X_train, y_train)

print("> Accuracy : ", reg.score(X_validation, y_validation))

predicton = reg.predict(X_test)
df_test = df_test.assign(hindex=predicton)

df_output = pd.concat([df_train, df_validation, df_test])
df_output = df_output.sort_index()

print(df_output["hindex"].describe())

df_output.to_csv(output_file_name, sep=',')

