import argparse
import pandas as pd
import progress_bar as pb

parser = argparse.ArgumentParser(description="Formats the coauthorship relations")
parser.add_argument("input_file", help="Input .edgelist file")
parser.add_argument("output_file", help="Output .csv file")

args = parser.parse_args()
    
input_file_name = args.input_file
output_file_name = args.output_file

pb.init(1, _prefix='Formating coauthorship \t \t:', _suffix='Complete')
pb.progress(0)

df_input = pd.read_csv(input_file_name, sep=" ", header=None)
df_input.columns=["author_1", "author_2"]
df_input.to_csv(output_file_name, sep=";", index=False)

pb.progress(1)
