import pandas as pd
import progress_bar as pb
import argparse


parser = argparse.ArgumentParser(
    description="Générer un vecteur à partir d'un modèle sauvegardé")
    
parser.add_argument("input_file", help="Input file to reindex")
parser.add_argument("dictionnary_file", help="Dictionnary file")
parser.add_argument("output_file", help="Output reindexed file")
parser.add_argument(
    "-column", help="Name of the column to reindex", type=str)
parser.add_argument(
    "-reverse", help="Reverse applying of the dictionnary or not", default=False, type=bool)

args = parser.parse_args()

input_file_name = args.input_file
dictionnary_file_name = args.dictionnary_file
output_file_name = args.output_file
column = args.column
reverse = args.reverse

# Init progress bar
pb.init(1, _prefix="Reindexing " + column + " \t \t", _suffix='Complete')
pb.progress(0)

df_dictionnary = pd.read_csv(dictionnary_file_name, sep=";")
df_dictionnary = df_dictionnary.rename(columns={"author":"author_dict", "new_author":"new_author_dict"})


if (reverse):
    df_input = pd.read_csv(input_file_name, sep=",")
    df_output = df_dictionnary.merge(df_input, left_on="new_author_dict", right_on=column)
    df_output = df_output.drop([column, "new_author_dict"], axis=1)
    df_output = df_output.rename(columns={"author_dict":column})

    df_output.to_csv(output_file_name, sep=",", index=False)
else:
    df_input = pd.read_csv(input_file_name, sep=";")
    df_output = df_dictionnary.merge(df_input, left_on="author_dict", right_on=column)
    df_output = df_output.drop([column, "author_dict"], axis=1)
    df_output = df_output.rename(columns={"new_author_dict":column})

    df_output.to_csv(output_file_name, sep=";", index=False)


pb.progress(1)
