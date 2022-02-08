import pandas as pd
import progress_bar as pb
import argparse

hindex_mean =  10.087608542191562
hindex_std =  12.588073

parser = argparse.ArgumentParser(
    description="Filters a submission to only leave entries from the test file")
    
parser.add_argument("input_file", help="Input file to filter")
parser.add_argument("test_file", help="Train file, has to be reindexed if input_file is reindexed and raw if input_file is raw")
parser.add_argument("output_file", help="Output filtered file")

args = parser.parse_args()

input_file_name = args.input_file
test_file_name = args.test_file
output_file_name = args.output_file

pb.init(1, _prefix="Filtering \t \t \t:", _suffix='Complete')
pb.progress(0)

df_input = pd.read_csv(input_file_name, sep=",")
df_test = pd.read_csv(test_file_name, sep=";")

row_count = df_input.shape[0]
pb.set_length(row_count)

df_output = df_test.merge(df_input, left_on="author", right_on="author", how="left", suffixes=["_test", ""])

df_output = df_output[["author", "hindex"]]

def compute_row_2(row):
    return[int(row.hindex)]

    
df_output = df_output.apply(lambda row: compute_row_2(row), result_type="expand", axis=1)
df_output = df_output.rename(columns={0: "hindex"})
df_output.insert(0, "author", df_test["author"])

df_output.to_csv(output_file_name, sep=',', index=False)

pb.progress(row_count)
