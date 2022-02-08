from sklearn.model_selection import train_test_split
import pandas as pd
import progress_bar as pb
import argparse

parser = argparse.ArgumentParser(description="Generating a test sample and a train sample from the global train sample")
parser.add_argument("input_file_train")
parser.add_argument("input_file_test")
parser.add_argument("output_file_train")
parser.add_argument("output_file_validation")
parser.add_argument("output_file_test")
parser.add_argument("-t", help="Validation proportion : if 0.01, 1% of the initial sample will end up in the validation sample", default=0.01, type=float)

args = parser.parse_args()
    
input_file_train_name = args.input_file_train
input_file_test_name = args.input_file_test
output_file_train_name = args.output_file_train
output_file_validation_name = args.output_file_validation
output_file_test_name = args.output_file_test
test_size = args.t

pb.init(4, _prefix = 'Generating samples \t \t', _suffix = 'Complete')
pb.progress(0)

df_input_train = pd.read_csv(input_file_train_name)
df_input_test = pd.read_csv(input_file_test_name, index_col=0)

pb.progress(1)

X_train, X_validation, y_train, y_validation = train_test_split(df_input_train["author"], df_input_train["hindex"], test_size=test_size)

pb.progress(2)

df_output_train = pd.DataFrame(data={"author" : X_train, "hindex" : y_train})
df_output_validation = pd.DataFrame(data={"author" : X_validation, "hindex" : y_validation})

pb.progress(3)

df_output_train.to_csv(output_file_train_name, sep=";", index=False)
df_output_validation.to_csv(output_file_validation_name, sep=";", index=False)
df_input_test.to_csv(output_file_test_name, sep=";", index=False)

pb.progress(4)
