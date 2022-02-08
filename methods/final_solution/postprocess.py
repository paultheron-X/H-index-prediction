import pandas as pd
import progress_bar as pb
import argparse

parser = argparse.ArgumentParser(
    description="Postprocessing of the output of the neural network")
    
parser.add_argument("input_file")
parser.add_argument("author_count_file")
parser.add_argument("test_file")
parser.add_argument("output_file")

args = parser.parse_args()

input_file_name = args.input_file
author_count_file_name = args.author_count_file
test_file_name = args.test_file
output_file_name = args.output_file

# Init progress bar
pb.init(1, _prefix="Filtering \t \t \t:", _suffix='Complete')
pb.progress(0)

df_input = pd.read_csv(input_file_name, sep=",")
df_author_count = pd.read_csv(author_count_file_name, sep=";", index_col=0)
df_test = pd.read_csv(test_file_name, sep=";")

row_count = df_input.shape[0]
pb.set_length(row_count)
count = 0

def max_hindex_to_abstract_count(row):
    global count
    pb.progress(count)
    count += 1
    try:
        max_hindex = df_author_count.loc[int(row.name)]
        if (max_hindex < 5):
            row.hindex = round(min(row.hindex, max_hindex))
        row.hindex = max(row.hindex, 1)
        return row
    except:
        row.hindex = round(row.hindex)
        row.hindex = max(row.hindex, 1)
        return row
        

df_input = df_input.apply(lambda row: max_hindex_to_abstract_count(row), axis=1)

df_output = df_test.merge(df_input, left_on="author", right_on="author", how="left", suffixes=["_test", ""])

df_output = df_output[["author", "hindex"]]

pb.progress(row_count)
print(df_output["hindex"].describe())

df_output.to_csv(output_file_name, sep=',', index=False)

