
import pandas as pd
import argparse


parser = argparse.ArgumentParser(
    description="...")
parser.add_argument("input")
parser.add_argument("output")

args = parser.parse_args()

input_edge_file_name = args.input
output_edge_file_name = args.output

df_input_edge = pd.read_csv(input_edge_file_name, sep=";")

df_edge_reverse = df_input_edge.rename(columns={"author_1":"author_2", "author_2":"author_1" })
df_output_edge = df_input_edge.append(df_edge_reverse, ignore_index=True)

df_output_edge.to_csv(output_edge_file_name, sep=";", index=False)