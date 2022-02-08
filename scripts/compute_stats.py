import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Compute the mean stats of the train data to know more about hindex")
parser.add_argument("input", "The input raw train file")
parser.add_argument("column")

args = parser.parse_args()
    
input_file_name = args.input
column = args.column

df_input_train = pd.read_csv(input_file_name)

print("> Output : ")
print(df_input_train[column].describe())
